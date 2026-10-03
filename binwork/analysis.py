"""Recursive-disassembly and control-flow recovery."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field, replace
from enum import Enum

from .address import AddressSpace
from .isa import BranchKind, Instruction, Operand, OperandKind, Operation, decode_instruction

_MAX_INSTRUCTION_BYTES = 4


def _unique_by(items, key):
    seen = set()
    out = []
    for item in items:
        value = key(item)
        if value not in seen:
            seen.add(value)
            out.append(item)
    return out


class TargetStatus(str, Enum):
    QUEUED = "queued"
    RESOLVED = "resolved"
    UNMAPPED = "unmapped"


class EdgeKind(str, Enum):
    FALLTHROUGH = "fallthrough"
    DIRECT_BRANCH = "direct_branch"
    CONDITION_TAKEN = "condition_taken"
    CONDITION_NOT_TAKEN = "condition_not_taken"
    CALL = "call"
    TAIL_CALL = "tail_call"
    CALL_RETURN = "call_return"
    INDIRECT = "indirect"


class EvidenceKind(str, Enum):
    DIRECT_OPERAND = "direct_operand"
    KNOWN_CONSTANT = "known_constant"
    JUMP_TABLE = "jump_table"
    TRUNCATED_JUMP_TABLE = "truncated_jump_table"
    CALL_OPERAND = "call_operand"
    TAIL_CALL = "tail_call"
    RETURN_INSTRUCTION = "return_instruction"
    CALL_RETURN_UNKNOWN = "call_return_unknown"


class CandidateStatus(str, Enum):
    CANDIDATE = "candidate"
    TRUNCATED = "truncated"


class PathStatus(str, Enum):
    CONFIRMED = "confirmed"
    ASSUMED = "assumed"
    UNRESOLVED = "unresolved"
    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    INSUFFICIENT = "insufficient"
    CONFLICT = "conflict"


class RevisionKind(str, Enum):
    PROCEDURE_RETURN = "procedure_return"
    JUMP_TABLE = "jump_table"
    EXPLORATION_BUDGET = "exploration_budget"
    CALL_BASIS = "call_basis"


@dataclass(frozen=True, slots=True)
class EdgeEvidence:
    kind: EvidenceKind
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Assumption:
    assumption_id: str
    kind: str
    subject_id: str
    description: str


@dataclass(frozen=True, slots=True)
class EntryValueConstraint:
    constraint_id: str
    entry_address: int
    register: int
    value: int


@dataclass(frozen=True, slots=True)
class BranchChoiceConstraint:
    constraint_id: str
    branch_address: int
    taken: bool
    target_address: int | None = None


@dataclass(frozen=True, slots=True)
class CallResultConstraint:
    constraint_id: str
    may_return: bool
    call_address: int | None = None
    callee_address: int | None = None


@dataclass(frozen=True, slots=True)
class CallBehaviorStatement:
    """One user-declared claim about how a call behaves.

    A statement can say whether the call returns, which register values hold
    after it returns, and which branch outcomes apply along the continuation.
    Statements are versioned through CallBehaviorBasis; withdrawing or
    contradicting a statement never rewrites conclusions drawn earlier.
    """

    statement_id: str
    call_address: int | None = None
    callee_address: int | None = None
    may_return: bool | None = None
    return_values: tuple[tuple[int, int], ...] = ()
    branch_choices: tuple[BranchChoiceConstraint, ...] = ()
    note: str = ""

    def matches_call(self, call_address: int, callee_address: int | None) -> bool:
        if self.call_address is not None and self.call_address != call_address:
            return False
        if self.callee_address is not None and self.callee_address != callee_address:
            return False
        return True


@dataclass(frozen=True, slots=True)
class BasisConflict:
    conflict_id: str
    statement_ids: tuple[str, ...]
    subject: str
    reason: str


@dataclass(frozen=True, slots=True)
class CallBehaviorBasis:
    """A versioned set of call-behavior statements used as query evidence."""

    basis_id: str
    version: int
    statements: tuple[CallBehaviorStatement, ...] = ()
    withdrawn_statement_ids: tuple[str, ...] = ()
    supersedes: str | None = None
    note: str = ""

    @property
    def ref(self) -> str:
        return f"{self.basis_id}@v{self.version}"

    def active_statements(self) -> tuple[CallBehaviorStatement, ...]:
        withdrawn = set(self.withdrawn_statement_ids)
        return tuple(
            statement
            for statement in self.statements
            if statement.statement_id not in withdrawn
        )

    def conflicts(self) -> tuple[BasisConflict, ...]:
        conflicts: list[BasisConflict] = []
        active = self.active_statements()
        may_return = [item for item in active if item.may_return is not None]
        for index, left in enumerate(may_return):
            for right in may_return[index + 1 :]:
                if left.may_return == right.may_return:
                    continue
                if (
                    left.call_address is not None
                    and right.call_address is not None
                    and left.call_address != right.call_address
                ):
                    continue
                if (
                    left.callee_address is not None
                    and right.callee_address is not None
                    and left.callee_address != right.callee_address
                ):
                    continue
                conflicts.append(
                    BasisConflict(
                        conflict_id=(
                            f"basis-conflict:{self.ref}:call:"
                            f"{left.statement_id}:{right.statement_id}"
                        ),
                        statement_ids=(left.statement_id, right.statement_id),
                        subject=(
                            f"call={left.call_address if left.call_address is not None else right.call_address},"
                            f"callee={left.callee_address if left.callee_address is not None else right.callee_address}"
                        ),
                        reason="statements disagree on whether the same call returns",
                    )
                )
        branch_groups: dict[int, list[tuple[CallBehaviorStatement, BranchChoiceConstraint]]] = {}
        for statement in active:
            for choice in statement.branch_choices:
                branch_groups.setdefault(choice.branch_address, []).append((statement, choice))
        for address, group in sorted(branch_groups.items()):
            if len({choice.taken for _, choice in group}) > 1:
                conflicts.append(
                    BasisConflict(
                        conflict_id=f"basis-conflict:{self.ref}:branch:{address:04x}",
                        statement_ids=tuple(
                            dict.fromkeys(statement.statement_id for statement, _ in group)
                        ),
                        subject=f"branch=0x{address:04x}",
                        reason="statements require opposite outcomes for the same branch",
                    )
                )
        return tuple(conflicts)

    def conflicting_statement_ids(self) -> frozenset[str]:
        return frozenset(
            statement_id
            for conflict in self.conflicts()
            for statement_id in conflict.statement_ids
        )

    def usable_statements(self) -> tuple[CallBehaviorStatement, ...]:
        blocked = self.conflicting_statement_ids()
        return tuple(
            statement
            for statement in self.active_statements()
            if statement.statement_id not in blocked
        )

    def derive(
        self,
        *,
        statements: tuple[CallBehaviorStatement, ...] | list[CallBehaviorStatement] | None = None,
        withdraw: tuple[str, ...] | list[str] = (),
        note: str = "",
    ) -> "CallBehaviorBasis":
        return CallBehaviorBasis(
            basis_id=self.basis_id,
            version=self.version + 1,
            statements=tuple(statements) if statements is not None else self.statements,
            withdrawn_statement_ids=self.withdrawn_statement_ids + tuple(withdraw),
            supersedes=self.ref,
            note=note,
        )

    def to_dict(self) -> dict:
        return {
            "basis_id": self.basis_id,
            "version": self.version,
            "statements": [
                {
                    "statement_id": item.statement_id,
                    "call_address": item.call_address,
                    "callee_address": item.callee_address,
                    "may_return": item.may_return,
                    "return_values": [list(pair) for pair in item.return_values],
                    "branch_choices": [asdict(choice) for choice in item.branch_choices],
                    "note": item.note,
                }
                for item in self.statements
            ],
            "withdrawn_statement_ids": list(self.withdrawn_statement_ids),
            "supersedes": self.supersedes,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CallBehaviorBasis":
        return cls(
            basis_id=data["basis_id"],
            version=data["version"],
            statements=tuple(
                CallBehaviorStatement(
                    statement_id=item["statement_id"],
                    call_address=item.get("call_address"),
                    callee_address=item.get("callee_address"),
                    may_return=item.get("may_return"),
                    return_values=tuple(
                        (int(pair[0]), int(pair[1])) for pair in item.get("return_values", ())
                    ),
                    branch_choices=tuple(
                        BranchChoiceConstraint(**choice)
                        for choice in item.get("branch_choices", ())
                    ),
                    note=item.get("note", ""),
                )
                for item in data.get("statements", ())
            ),
            withdrawn_statement_ids=tuple(data.get("withdrawn_statement_ids", ())),
            supersedes=data.get("supersedes"),
            note=data.get("note", ""),
        )


@dataclass(frozen=True, slots=True)
class BasisUsage:
    """Which basis version a query conclusion actually relied on."""

    basis_id: str
    basis_version: int
    statement_ids: tuple[str, ...] = ()

    @property
    def ref(self) -> str:
        return f"{self.basis_id}@v{self.basis_version}"


@dataclass(frozen=True, slots=True)
class PathConstraints:
    entry_values: tuple[EntryValueConstraint, ...] = ()
    branch_choices: tuple[BranchChoiceConstraint, ...] = ()
    call_results: tuple[CallResultConstraint, ...] = ()

    def ids(self) -> tuple[str, ...]:
        return (
            tuple(item.constraint_id for item in self.entry_values)
            + tuple(item.constraint_id for item in self.branch_choices)
            + tuple(item.constraint_id for item in self.call_results)
        )


@dataclass(frozen=True, slots=True)
class ConstraintConflict:
    conflict_id: str
    constraint_ids: tuple[str, ...]
    subject: str
    reason: str
    evidence: tuple[EdgeEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class MissingBehavior:
    requirement_id: str
    kind: str
    subject_address: int
    description: str
    relation_id: str | None = None
    constraint_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExcludedRelation:
    relation_id: str
    source_address: int
    target_address: int | None
    reason: str
    violated_constraint_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UnfinishedScope:
    scope_id: str
    kind: str
    address: int
    description: str
    relation_id: str | None = None
    evidence: tuple[EdgeEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class PathStep:
    relation_id: str
    source_instruction_id: str | None
    source_address: int
    target_instruction_id: str | None
    target_address: int | None
    kind: EdgeKind
    confirmed: bool
    assumption_ids: tuple[str, ...] = ()
    evidence: tuple[EdgeEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class ReachablePath:
    path_id: str
    source_address: int
    target_address: int
    status: PathStatus
    steps: tuple[PathStep, ...]
    assumption_ids: tuple[str, ...] = ()
    target_pending: bool = False
    constraint_ids: tuple[str, ...] = ()
    register_values: tuple[tuple[int, int], ...] = ()
    used_constraint_ids: tuple[str, ...] = ()
    basis_statement_ids: tuple[str, ...] = ()

    @property
    def evidence(self) -> tuple[EdgeEvidence, ...]:
        return tuple(item for step in self.steps for item in step.evidence)


@dataclass(frozen=True, slots=True)
class PathQueryResult:
    source_address: int | None
    target_address: int
    paths: tuple[ReachablePath, ...]
    query_id: str = "query:unconstrained"
    constraints: PathConstraints = field(default_factory=PathConstraints)
    conflicts: tuple[ConstraintConflict, ...] = ()
    missing_behaviors: tuple[MissingBehavior, ...] = ()
    excluded_relations: tuple[ExcludedRelation, ...] = ()
    unfinished_scopes: tuple[UnfinishedScope, ...] = ()
    basis_usage: BasisUsage | None = None

    @property
    def status(self) -> PathStatus:
        if self.conflicts:
            return PathStatus.CONFLICT
        if any(path.status is PathStatus.CONFIRMED for path in self.paths):
            return PathStatus.REACHABLE if self.constraints.ids() else PathStatus.CONFIRMED
        if any(path.status is PathStatus.ASSUMED for path in self.paths):
            return PathStatus.ASSUMED
        if self.missing_behaviors or self.unfinished_scopes:
            return PathStatus.INSUFFICIENT
        if self.constraints.ids():
            return PathStatus.UNREACHABLE
        return PathStatus.UNRESOLVED


@dataclass(frozen=True, slots=True)
class ConstraintDifference:
    relation_id: str
    source_address: int
    target_address: int | None
    left_included: bool
    right_included: bool
    reason: str
    constraint_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConditionComparison:
    target_address: int
    left: PathQueryResult
    right: PathQueryResult
    differences: tuple[ConstraintDifference, ...]


@dataclass(frozen=True, slots=True)
class CallBehaviorComparison:
    target_address: int
    left: PathQueryResult
    right: PathQueryResult
    differences: tuple[ConstraintDifference, ...]


@dataclass(frozen=True, slots=True)
class CallBehaviorRevision:
    """Preview of how moving from one basis version to another shifts a query.

    The previous query result is kept untouched; the derived query is a new
    object with its own query_id. Confirmed paths that never used a changed
    or withdrawn statement stay valid and are listed under preserved_paths.
    """

    revision_id: str
    target_address: int
    source_address: int | None
    previous_basis: CallBehaviorBasis
    basis: CallBehaviorBasis
    previous_query: PathQueryResult
    derived_query: PathQueryResult
    changed_statement_ids: tuple[str, ...]
    withdrawn_statement_ids: tuple[str, ...]
    added_statement_ids: tuple[str, ...]
    affected_paths: tuple[ReachablePath, ...]
    preserved_paths: tuple[ReachablePath, ...]
    affected_candidates: tuple[IndirectTargetCandidate, ...]
    affected_missing_behaviors: tuple[MissingBehavior, ...]
    affected_unfinished_scopes: tuple[UnfinishedScope, ...]
    detail: str = ""


@dataclass(frozen=True, slots=True)
class AnalysisRevision:
    revision_id: str
    kind: RevisionKind
    subject_address: int
    changed_relationship_ids: tuple[str, ...]
    detail: str = ""


@dataclass(frozen=True, slots=True)
class RevisionReport:
    previous_result: AnalysisResult
    result: AnalysisResult
    revision: AnalysisRevision


@dataclass(frozen=True, slots=True)
class IndirectTargetCandidate:
    candidate_id: str
    source_instruction_id: str
    target_address: int
    status: CandidateStatus
    evidence: tuple[EdgeEvidence, ...]


@dataclass(frozen=True, slots=True)
class PendingTarget:
    target_id: str
    address: int
    status: TargetStatus
    evidence: tuple[EdgeEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class KnownIndirectTarget:
    source_address: int
    register: int
    target_address: int


@dataclass(frozen=True, slots=True)
class JumpTableConstraint:
    table_address: int
    registers: tuple[int, ...] = ()
    source_addresses: tuple[int, ...] = ()
    entry_count: int | None = None
    max_entries: int = 8
    entries: tuple[int, ...] | None = None

    def matches(self, instruction: Instruction) -> bool:
        if self.source_addresses and instruction.address not in self.source_addresses:
            return False
        register = instruction.operands[0].value if instruction.operands else None
        if self.registers and register not in self.registers:
            return False
        return True


@dataclass(frozen=True, slots=True)
class AnalysisBudget:
    max_instructions: int | None = None


@dataclass(frozen=True, slots=True)
class Procedure:
    procedure_id: str
    entry_address: int
    may_return: bool | None


@dataclass(frozen=True, slots=True)
class CallRelationship:
    call_instruction_id: str
    callee_address: int
    return_address: int | None
    tail_call: bool
    callee_may_return: bool | None
    evidence: tuple[EdgeEvidence, ...]


@dataclass(frozen=True, slots=True)
class CfgEdge:
    edge_id: str
    source_instruction_id: str
    kind: EdgeKind
    target_address: int | None = None
    target_block_id: str | None = None
    resolved: bool = True
    evidence: tuple[EdgeEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class BasicBlock:
    block_id: str
    start_address: int
    end_address: int
    instruction_ids: tuple[str, ...]
    outgoing_edge_ids: tuple[str, ...]
    overlap: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    entry_points: tuple[int, ...]
    instructions: tuple[Instruction, ...]
    edges: tuple[CfgEdge, ...]
    blocks: tuple[BasicBlock, ...]
    target_statuses: dict[int, TargetStatus]
    unresolved_indirect_jumps: tuple[str, ...]
    roots: tuple[int, ...]
    indirect_target_candidates: tuple[IndirectTargetCandidate, ...] = ()
    procedures: tuple[Procedure, ...] = ()
    call_relationships: tuple[CallRelationship, ...] = ()
    budget_exhausted: bool = False
    pending_targets: tuple[PendingTarget, ...] = ()
    revisions: tuple[AnalysisRevision, ...] = ()
    call_basis: CallBehaviorBasis | None = None

    def instruction(self, instruction_id: str) -> Instruction:
        for item in self.instructions:
            if item.instruction_id == instruction_id:
                return item
        raise KeyError(instruction_id)

    def block_for_instruction(self, instruction_id: str) -> BasicBlock:
        for block in self.blocks:
            if instruction_id in block.instruction_ids:
                return block
        raise KeyError(instruction_id)

    def block_outgoing(self, block: BasicBlock) -> tuple[CfgEdge, ...]:
        last_id = block.instruction_ids[-1]
        return tuple(edge for edge in self.edges if edge.source_instruction_id == last_id)

    def disassembled_text(self) -> str:
        return "\n".join(item.render() for item in self.instructions)

    def confirmed_edges(self) -> tuple[CfgEdge, ...]:
        return tuple(edge for edge in self.edges if edge.resolved)

    def candidates_for(self, instruction_id: str) -> tuple[IndirectTargetCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.indirect_target_candidates
            if candidate.source_instruction_id == instruction_id
        )

    def assumptions(self) -> tuple[Assumption, ...]:
        return tuple(
            Assumption(
                assumption_id=f"assume:{candidate.candidate_id}",
                kind=candidate.status.value,
                subject_id=candidate.candidate_id,
                description=f"indirect jump at 0x{self.instruction(candidate.source_instruction_id).address:04x} reaches 0x{candidate.target_address:04x}",
            )
            for candidate in self.indirect_target_candidates
        )

    def assumption_map(self) -> dict[str, Assumption]:
        return {item.assumption_id: item for item in self.assumptions()}

    def paths_to(
        self,
        target_address: int,
        *,
        source_address: int | None = None,
        constraints: PathConstraints | None = None,
        entry_values: tuple[EntryValueConstraint, ...] = (),
        branch_choices: tuple[BranchChoiceConstraint, ...] = (),
        call_results: tuple[CallResultConstraint, ...] = (),
        basis: CallBehaviorBasis | None = None,
        include_statuses: frozenset[PathStatus] | set[PathStatus] | tuple[PathStatus, ...] = (
            PathStatus.CONFIRMED,
            PathStatus.ASSUMED,
            PathStatus.UNRESOLVED,
        ),
        max_paths: int = 32,
    ) -> PathQueryResult:
        origins = (source_address,) if source_address is not None else self.entry_points
        conditions = constraints or PathConstraints(
            entry_values=tuple(entry_values),
            branch_choices=tuple(branch_choices),
            call_results=tuple(call_results),
        )
        basis = basis if basis is not None else self.call_basis
        usable_statements: tuple[CallBehaviorStatement, ...] = ()
        basis_conflicts: tuple[BasisConflict, ...] = ()
        if basis is not None:
            usable_statements = basis.usable_statements()
            basis_conflicts = basis.conflicts()
            conditions = replace(
                conditions,
                branch_choices=conditions.branch_choices
                + tuple(
                    replace(
                        choice,
                        constraint_id=f"{statement.statement_id}:branch:{choice.branch_address:04x}",
                    )
                    for statement in usable_statements
                    for choice in statement.branch_choices
                ),
                call_results=conditions.call_results
                + tuple(
                    CallResultConstraint(
                        statement.statement_id,
                        statement.may_return,
                        statement.call_address,
                        statement.callee_address,
                    )
                    for statement in usable_statements
                    if statement.may_return is not None
                ),
            )
        conflicts = self._constraint_conflicts(conditions, origins)
        conflicts.extend(
            ConstraintConflict(
                conflict_id=item.conflict_id,
                constraint_ids=item.statement_ids,
                subject=item.subject,
                reason=item.reason,
            )
            for item in basis_conflicts
        )
        query_id = self._query_id(target_address, source_address, conditions, basis)
        paths: list[ReachablePath] = []
        wanted = frozenset(include_statuses)
        excluded: list[ExcludedRelation] = []
        missing: list[MissingBehavior] = []
        unfinished: list[UnfinishedScope] = []
        if not conflicts:
            for origin in origins:
                if PathStatus.CONFIRMED in wanted:
                    found, scope = self._confirmed_paths(
                        origin,
                        target_address,
                        max_paths - len(paths),
                        conditions,
                        usable_statements,
                    )
                    paths.extend(found)
                    missing.extend(scope)
                if PathStatus.ASSUMED in wanted:
                    found, scope = self._assumed_paths(
                        origin,
                        target_address,
                        max_paths - len(paths),
                        conditions,
                        usable_statements,
                    )
                    paths.extend(found)
                    missing.extend(scope)
                if len(paths) >= max_paths:
                    break
            excluded, missing, unfinished = self._condition_frontier(
                target_address,
                origins,
                conditions,
                tuple(dict.fromkeys(item.requirement_id for item in missing)),
            )
        if not paths and target_address in origins and not conflicts:
            paths.append(
                ReachablePath(
                    path_id=f"path:0:{origin_to_id(origins, target_address)}:{target_address:04x}",
                    source_address=target_address,
                    target_address=target_address,
                    status=PathStatus.CONFIRMED,
                    steps=(),
                    constraint_ids=conditions.ids(),
                )
            )
        if (
            not paths
            and not (missing or unfinished)
            and not conflicts
            and (PathStatus.UNRESOLVED in wanted or PathStatus.INSUFFICIENT in wanted or conditions.ids())
        ):
            source = source_address if source_address is not None else (origins[0] if origins else None)
            if source is not None:
                paths.append(
                    ReachablePath(
                        path_id=f"path:unresolved:{source:04x}:{target_address:04x}",
                        source_address=source,
                        target_address=target_address,
                        status=PathStatus.UNRESOLVED,
                        steps=(),
                        target_pending=any(
                            item.address == target_address for item in self.pending_targets
                        ),
                        constraint_ids=conditions.ids(),
                    )
                )
        basis_usage: BasisUsage | None = None
        if basis is not None:
            known_statement_ids = {item.statement_id for item in basis.statements}
            attributed: list[ReachablePath] = []
            for path in paths:
                statement_ids = tuple(
                    dict.fromkeys(
                        used.split(":branch:", 1)[0]
                        for used in path.used_constraint_ids
                        if used.split(":branch:", 1)[0] in known_statement_ids
                    )
                )
                attributed.append(replace(path, basis_statement_ids=statement_ids))
            paths = attributed
            basis_usage = BasisUsage(
                basis.basis_id,
                basis.version,
                tuple(
                    dict.fromkeys(
                        statement_id
                        for path in paths
                        for statement_id in path.basis_statement_ids
                    )
                ),
            )
        return PathQueryResult(
            source_address,
            target_address,
            tuple(paths[:max_paths]),
            query_id=query_id,
            constraints=conditions,
            conflicts=tuple(conflicts),
            missing_behaviors=tuple(_unique_by(missing, lambda item: item.requirement_id)),
            excluded_relations=tuple(_unique_by(excluded, lambda item: item.relation_id + item.reason)),
            unfinished_scopes=tuple(_unique_by(unfinished, lambda item: item.scope_id)),
            basis_usage=basis_usage,
        )

    def compare_conditions(
        self,
        target_address: int,
        left: PathConstraints,
        right: PathConstraints,
        *,
        source_address: int | None = None,
    ) -> ConditionComparison:
        left_result = self.paths_to(
            target_address,
            source_address=source_address,
            constraints=left,
        )
        right_result = self.paths_to(
            target_address,
            source_address=source_address,
            constraints=right,
        )
        return ConditionComparison(
            target_address,
            left_result,
            right_result,
            self._relation_differences(left_result, right_result),
        )

    def compare_call_behaviors(
        self,
        target_address: int,
        left_basis: CallBehaviorBasis,
        right_basis: CallBehaviorBasis,
        *,
        source_address: int | None = None,
        constraints: PathConstraints | None = None,
    ) -> CallBehaviorComparison:
        left_result = self.paths_to(
            target_address,
            source_address=source_address,
            constraints=constraints,
            basis=left_basis,
        )
        right_result = self.paths_to(
            target_address,
            source_address=source_address,
            constraints=constraints,
            basis=right_basis,
        )
        return CallBehaviorComparison(
            target_address,
            left_result,
            right_result,
            self._relation_differences(left_result, right_result),
        )

    def revise_call_behavior(
        self,
        target_address: int,
        previous_basis: CallBehaviorBasis,
        basis: CallBehaviorBasis,
        *,
        source_address: int | None = None,
        constraints: PathConstraints | None = None,
    ) -> CallBehaviorRevision:
        previous_query = self.paths_to(
            target_address,
            source_address=source_address,
            constraints=constraints,
            basis=previous_basis,
        )
        derived_query = self.paths_to(
            target_address,
            source_address=source_address,
            constraints=constraints,
            basis=basis,
        )
        old_statements = {item.statement_id: item for item in previous_basis.statements}
        new_statements = {item.statement_id: item for item in basis.statements}
        newly_withdrawn = tuple(
            statement_id
            for statement_id in basis.withdrawn_statement_ids
            if statement_id not in previous_basis.withdrawn_statement_ids
        )
        changed = tuple(
            statement_id
            for statement_id, old in old_statements.items()
            if statement_id in new_statements
            and new_statements[statement_id] != old
            and statement_id not in newly_withdrawn
        )
        removed = tuple(
            statement_id
            for statement_id in old_statements
            if statement_id not in new_statements
        )
        added = tuple(
            statement_id
            for statement_id in new_statements
            if statement_id not in old_statements
        )
        invalidated = frozenset(changed + removed + newly_withdrawn)
        affected_paths = tuple(
            path
            for path in previous_query.paths
            if invalidated & frozenset(path.basis_statement_ids)
        )
        preserved_paths = tuple(
            path
            for path in previous_query.paths
            if path.status is PathStatus.CONFIRMED
            and not (invalidated & frozenset(path.basis_statement_ids))
        )
        affected_relation_ids = {
            step.relation_id for path in affected_paths for step in path.steps
        }
        affected_addresses = {
            step.source_address for path in affected_paths for step in path.steps
        }
        affected_candidates = tuple(
            candidate
            for candidate in self.indirect_target_candidates
            if candidate.candidate_id in affected_relation_ids
            or self.instruction(candidate.source_instruction_id).address in affected_addresses
        )
        changed_subjects = {
            value
            for statement_id in invalidated
            if statement_id in old_statements
            for value in (
                old_statements[statement_id].call_address,
                old_statements[statement_id].callee_address,
            )
            if value is not None
        }
        affected_missing = tuple(
            item
            for item in previous_query.missing_behaviors
            if item.subject_address in changed_subjects
            or (item.relation_id is not None and item.relation_id in affected_relation_ids)
        )
        affected_unfinished = (
            previous_query.unfinished_scopes
            if (affected_paths or affected_missing)
            else ()
        )
        return CallBehaviorRevision(
            revision_id=f"basis-revision:{basis.ref}:{target_address:04x}",
            target_address=target_address,
            source_address=source_address,
            previous_basis=previous_basis,
            basis=basis,
            previous_query=previous_query,
            derived_query=derived_query,
            changed_statement_ids=changed,
            withdrawn_statement_ids=newly_withdrawn,
            added_statement_ids=added,
            affected_paths=affected_paths,
            preserved_paths=preserved_paths,
            affected_candidates=affected_candidates,
            affected_missing_behaviors=affected_missing,
            affected_unfinished_scopes=affected_unfinished,
            detail=basis.note,
        )

    def _relation_differences(
        self,
        left_result: PathQueryResult,
        right_result: PathQueryResult,
    ) -> tuple[ConstraintDifference, ...]:
        included_left = {
            (step.relation_id, step.target_address)
            for path in left_result.paths
            for step in path.steps
        }
        included_right = {
            (step.relation_id, step.target_address)
            for path in right_result.paths
            for step in path.steps
        }
        differences: list[ConstraintDifference] = []
        all_relations: dict[tuple[str, int | None], tuple[int, int | None, str, tuple[str, ...]]] = {}
        for index, result in enumerate((left_result, right_result)):
            for item in result.excluded_relations:
                key = (item.relation_id, item.target_address)
                current = all_relations.get(key)
                if current is None:
                    all_relations[key] = (item.source_address, item.target_address, item.reason, item.violated_constraint_ids)
            for path in result.paths:
                for step in path.steps:
                    key = (step.relation_id, step.target_address)
                    all_relations.setdefault(
                        key,
                        (step.source_address, step.target_address, "", ()),
                    )
        for (relation_id, target_address), (source_address, _, reason, constraint_ids) in all_relations.items():
            key = (relation_id, target_address)
            in_left = key in included_left
            in_right = key in included_right
            if in_left != in_right:
                differences.append(
                    ConstraintDifference(
                        relation_id=relation_id,
                        source_address=source_address,
                        target_address=target_address,
                        left_included=in_left,
                        right_included=in_right,
                        reason=reason or "path witness is available only under this set of conditions",
                        constraint_ids=constraint_ids,
                    )
                )
        return tuple(differences)

    def _confirmed_paths(
        self,
        source: int,
        target: int,
        limit: int,
        conditions: PathConstraints,
        call_effects: tuple[CallBehaviorStatement, ...] = (),
    ) -> tuple[list[ReachablePath], list[MissingBehavior]]:
        return self._search_paths(
            source,
            target,
            limit,
            conditions=conditions,
            use_candidates=False,
            call_effects=call_effects,
        )

    def _assumed_paths(
        self,
        source: int,
        target: int,
        limit: int,
        conditions: PathConstraints,
        call_effects: tuple[CallBehaviorStatement, ...] = (),
    ) -> tuple[list[ReachablePath], list[MissingBehavior]]:
        found, missing = self._search_paths(
            source,
            target,
            limit,
            conditions=conditions,
            use_candidates=True,
            call_effects=call_effects,
        )
        return [
            path
            for path in found
            if path.status is PathStatus.ASSUMED
        ], missing

    def _search_paths(
        self,
        source: int,
        target: int,
        limit: int,
        *,
        use_candidates: bool,
        conditions: PathConstraints,
        call_effects: tuple[CallBehaviorStatement, ...] = (),
    ) -> tuple[list[ReachablePath], list[MissingBehavior]]:
        if limit <= 0:
            return []
        outgoing: dict[int, list[PathStep]] = {}
        instructions_by_id = {item.instruction_id: item for item in self.instructions}
        records = self._address_records()

        def add_step(step: PathStep) -> None:
            outgoing.setdefault(step.source_address, []).append(step)

        for edge in self.edges:
            if edge.target_address is None:
                continue
            instruction = instructions_by_id.get(edge.source_instruction_id)
            if instruction is None:
                continue
            forced_return = self._forced_call_result(instruction, edge, conditions)
            if edge.kind is EdgeKind.CALL_RETURN and not edge.resolved:
                if forced_return is True:
                    target_records = records.get(edge.target_address, ())
                    step = PathStep(
                        relation_id=edge.edge_id,
                        source_instruction_id=edge.source_instruction_id,
                        source_address=instruction.address,
                        target_instruction_id=target_records[0] if target_records else None,
                        target_address=edge.target_address,
                        kind=edge.kind,
                        confirmed=False,
                        evidence=edge.evidence,
                    )
                    add_step(step)
                continue
            if not edge.resolved:
                continue
            target_records = records.get(edge.target_address, ())
            add_step(
                PathStep(
                    relation_id=edge.edge_id,
                    source_instruction_id=edge.source_instruction_id,
                    source_address=instruction.address,
                    target_instruction_id=target_records[0] if target_records else None,
                    target_address=edge.target_address,
                    kind=edge.kind,
                    confirmed=True,
                    evidence=edge.evidence,
                )
            )

        if use_candidates:
            for candidate in self.indirect_target_candidates:
                instruction = instructions_by_id.get(candidate.source_instruction_id)
                if instruction is None:
                    continue
                assumption_id = f"assume:{candidate.candidate_id}"
                target_records = records.get(candidate.target_address, ())
                add_step(
                    PathStep(
                        relation_id=candidate.candidate_id,
                        source_instruction_id=candidate.source_instruction_id,
                        source_address=instruction.address,
                        target_instruction_id=target_records[0] if target_records else None,
                        target_address=candidate.target_address,
                        kind=EdgeKind.INDIRECT,
                        confirmed=False,
                        assumption_ids=(assumption_id,),
                        evidence=candidate.evidence,
                    )
                )
        results: list[ReachablePath] = []
        initial_registers = {
            item.register: item.value
            for item in conditions.entry_values
            if item.entry_address == source
        }

        def walk(
            address: int,
            steps: list[PathStep],
            relations: set[str],
            assumed: set[str],
            visiting: set[int],
            registers: dict[int, int],
            active_constraints: set[str],
        ) -> None:
            if len(results) >= limit:
                return
            if address == target and (steps or source == target):
                results.append(
                    ReachablePath(
                        path_id=f"path:{source:04x}:{target:04x}:{len(results) + 1:03d}",
                        source_address=source,
                        target_address=target,
                        status=(
                            PathStatus.ASSUMED
                            if assumed or any(not step.confirmed for step in steps)
                            else PathStatus.CONFIRMED
                        ),
                        steps=tuple(steps),
                        assumption_ids=tuple(assumed),
                        target_pending=any(item.address == target for item in self.pending_targets),
                        constraint_ids=conditions.ids(),
                        register_values=tuple(sorted(registers.items())),
                        used_constraint_ids=tuple(sorted(active_constraints)),
                    )
                )
                return
            if address in visiting:
                return
            visiting.add(address)
            instruction_id = records.get(address, (None,))[0]
            instruction = instructions_by_id.get(instruction_id) if instruction_id is not None else None
            for step in outgoing.get(address, ()):
                if step.relation_id in relations:
                    continue
                if instruction is not None:
                    allowed, used, reason = self._step_allowed(
                        instruction,
                        step,
                        registers,
                        conditions,
                    )
                    if not allowed:
                        continue
                    next_registers = self._update_registers(instruction, registers, step)
                    if step.kind is EdgeKind.CALL_RETURN:
                        for statement in call_effects:
                            if statement.return_values and statement.matches_call(
                                instruction.address, instruction.branch_target
                            ):
                                next_registers.update(dict(statement.return_values))
                else:
                    used = ()
                    next_registers = dict(registers)
                relations.add(step.relation_id)
                added = [
                    assumption_id
                    for assumption_id in step.assumption_ids
                    if assumption_id not in assumed
                ]
                assumed.update(added)
                active_constraints.update(used)
                steps.append(step)
                walk(
                    step.target_address or address,
                    steps,
                    relations,
                    assumed,
                    visiting,
                    next_registers,
                    active_constraints,
                )
                steps.pop()
                for constraint_id in used:
                    active_constraints.discard(constraint_id)
                for assumption_id in added:
                    assumed.discard(assumption_id)
                relations.discard(step.relation_id)
                if len(results) >= limit:
                    break
            visiting.discard(address)

        walk(source, [], set(), set(), set(), dict(initial_registers), set())
        return results, []

    def _address_records(self) -> dict[int, tuple[str, ...]]:
        records: dict[int, list[str]] = {}
        for instruction in self.instructions:
            records.setdefault(instruction.address, []).append(instruction.instruction_id)
        return {address: tuple(ids) for address, ids in records.items()}

    def _query_id(
        self,
        target: int,
        source: int | None,
        conditions: PathConstraints,
        basis: CallBehaviorBasis | None = None,
    ) -> str:
        if not conditions.ids() and basis is None:
            return "query:unconstrained"
        payload = json.dumps(
            {
                "source": source,
                "target": target,
                "entry_values": [asdict(item) for item in conditions.entry_values],
                "branch_choices": [asdict(item) for item in conditions.branch_choices],
                "call_results": [asdict(item) for item in conditions.call_results],
                "basis": (
                    {
                        "ref": basis.ref,
                        "statements": [
                            {
                                "statement_id": item.statement_id,
                                "call_address": item.call_address,
                                "callee_address": item.callee_address,
                                "may_return": item.may_return,
                                "return_values": [list(pair) for pair in item.return_values],
                            }
                            for item in basis.active_statements()
                        ],
                    }
                    if basis is not None
                    else None
                ),
            },
            sort_keys=True,
        ).encode()
        return "query:" + hashlib.sha256(payload).hexdigest()[:16]

    def _known_register_at_branch(
        self,
        branch_address: int,
        register: int | None,
        origins: tuple[int, ...],
        conditions: PathConstraints,
    ) -> int | None:
        if register is None:
            return None
        records = self._address_records()
        instructions_by_id = {item.instruction_id: item for item in self.instructions}
        values: list[int] = []

        def walk(address: int, registers: dict[int, int], visiting: set[int]) -> None:
            if address == branch_address:
                if register in registers:
                    values.append(registers[register])
                return
            if address in visiting:
                return
            instruction_id = records.get(address, (None,))[0]
            instruction = instructions_by_id.get(instruction_id) if instruction_id is not None else None
            if instruction is None:
                return
            updated = self._update_registers(instruction, registers, None)  # type: ignore[arg-type]
            visiting.add(address)
            for edge in self.edges:
                if edge.source_instruction_id != instruction_id:
                    continue
                if edge.kind in {EdgeKind.CONDITION_TAKEN, EdgeKind.CONDITION_NOT_TAKEN}:
                    continue
                if not edge.resolved or edge.target_address is None:
                    continue
                walk(edge.target_address, updated, visiting)
            visiting.discard(address)

        for origin in origins:
            walk(
                origin,
                {
                    item.register: item.value
                    for item in conditions.entry_values
                    if item.entry_address == origin
                },
                set(),
            )
        unique = set(values)
        return values[0] if len(unique) == 1 else None

    def _constraint_conflicts(
        self,
        conditions: PathConstraints,
        origins: tuple[int, ...],
    ) -> list[ConstraintConflict]:
        conflicts: list[ConstraintConflict] = []
        entry_groups: dict[tuple[int, int], list[EntryValueConstraint]] = {}
        for item in conditions.entry_values:
            entry_groups.setdefault((item.entry_address, item.register), []).append(item)
        for (entry, register), group in entry_groups.items():
            values = {item.value for item in group}
            if len(values) > 1:
                conflicts.append(
                    ConstraintConflict(
                        conflict_id=f"conflict:entry:{entry:04x}:r{register}",
                        constraint_ids=tuple(item.constraint_id for item in group),
                        subject=f"entry=0x{entry:04x},register=r{register}",
                        reason="same entry register has different required values",
                    )
                )
        branch_groups: dict[int, list[BranchChoiceConstraint]] = {}
        for item in conditions.branch_choices:
            branch_groups.setdefault(item.branch_address, []).append(item)
        for address, group in branch_groups.items():
            instruction = next(
                (item for item in self.instructions if item.address == address),
                None,
            )
            if len({item.taken for item in group}) > 1:
                conflicts.append(
                    ConstraintConflict(
                        conflict_id=f"conflict:branch:{address:04x}",
                        constraint_ids=tuple(item.constraint_id for item in group),
                        subject=f"branch=0x{address:04x}",
                        reason="same conditional branch is required both taken and not taken",
                        evidence=instruction and (EdgeEvidence(EvidenceKind.DIRECT_OPERAND),) or (),
                    )
                )
            if instruction is not None and instruction.branch_kind is BranchKind.CONDITIONAL:
                register = instruction.operands[0].value if instruction.operands else None
                for item in group:
                    if instruction.operation is Operation.JZ:
                        required_zero = item.taken
                    elif instruction.operation is Operation.JNZ:
                        required_zero = not item.taken
                    else:
                        continue
                    known = self._known_register_at_branch(
                        address,
                        register,
                        origins,
                        conditions,
                    )
                    if known is not None and (known == 0) != required_zero:
                        conflicts.append(
                            ConstraintConflict(
                                conflict_id=f"conflict:entry-branch:{address:04x}:r{register}",
                                constraint_ids=(
                                    item.constraint_id,
                                    *(
                                        entry.constraint_id
                                        for entry in conditions.entry_values
                                        if entry.entry_address in origins and entry.register == register
                                    ),
                                ),
                                subject=f"branch=0x{address:04x},register=r{register}",
                                reason="entry value forces the opposite conditional branch outcome",
                                evidence=(EdgeEvidence(EvidenceKind.DIRECT_OPERAND),),
                            )
                        )
            for item in group:
                if item.target_address is not None:
                    actual = instruction.branch_target if instruction is not None else None
                    if item.taken and actual is not None and item.target_address != actual:
                        conflicts.append(
                            ConstraintConflict(
                                conflict_id=f"conflict:branch-target:{address:04x}:{item.target_address:04x}",
                                constraint_ids=(item.constraint_id,),
                                subject=f"branch=0x{address:04x}",
                                reason=f"taken edge targets 0x{actual:04x}, not 0x{item.target_address:04x}",
                            )
                        )
        call_groups: dict[tuple[int | None, int | None], list[CallResultConstraint]] = {}
        for item in conditions.call_results:
            call_groups.setdefault((item.call_address, item.callee_address), []).append(item)
        for key, group in call_groups.items():
            if len({item.may_return for item in group}) > 1:
                conflicts.append(
                    ConstraintConflict(
                        conflict_id=f"conflict:call:{key[0] if key[0] is not None else -1:04x}:{key[1] if key[1] is not None else -1:04x}",
                        constraint_ids=tuple(item.constraint_id for item in group),
                        subject=f"call=0x{key[0] or -1:04x},callee=0x{key[1] or -1:04x}",
                        reason="same call is required both to return and not to return",
                    )
                )
        for relationship in self.call_relationships:
            if relationship.callee_may_return is None:
                continue
            call = self.instruction(relationship.call_instruction_id)
            matching = [
                item
                for item in conditions.call_results
                if (item.call_address is None or item.call_address == call.address)
                and (item.callee_address is None or item.callee_address == relationship.callee_address)
            ]
            for item in matching:
                if item.may_return != relationship.callee_may_return:
                    conflicts.append(
                        ConstraintConflict(
                            conflict_id=f"conflict:call-behavior:{call.address:04x}:{relationship.callee_address:04x}",
                            constraint_ids=(item.constraint_id,),
                            subject=f"call=0x{call.address:04x},callee=0x{relationship.callee_address:04x}",
                            reason=(
                                "condition requires return but procedure is known not to return"
                                if item.may_return
                                else "condition requires no return but procedure is known to return"
                            ),
                            evidence=relationship.evidence,
                        )
                    )
        if conditions.entry_values and origins:
            for item in conditions.entry_values:
                if item.entry_address not in origins:
                    conflicts.append(
                        ConstraintConflict(
                            conflict_id=f"conflict:entry-source:{item.entry_address:04x}",
                            constraint_ids=(item.constraint_id,),
                            subject=f"entry=0x{item.entry_address:04x}",
                            reason="entry value is not for any queried entry point",
                        )
                    )
        return conflicts

    def _step_allowed(
        self,
        instruction: Instruction,
        step: PathStep,
        registers: dict[int, int],
        conditions: PathConstraints,
    ) -> tuple[bool, tuple[str, ...], str]:
        if instruction.branch_kind is BranchKind.CONDITIONAL:
            choice = next(
                (item for item in conditions.branch_choices if item.branch_address == instruction.address),
                None,
            )
            if choice is not None:
                is_taken = step.kind is EdgeKind.CONDITION_TAKEN
                if is_taken != choice.taken:
                    return False, (), "branch choice constraint selected the other edge"
                if choice.target_address is not None and is_taken and step.target_address != choice.target_address:
                    return False, (), "branch target constraint selected a different edge"
                return True, (choice.constraint_id,), "branch choice constraint selected this edge"
            register = instruction.operands[0].value if instruction.operands else None
            if register is not None and register in registers:
                if instruction.operation is Operation.JZ:
                    zero = registers[register] == 0
                    if (step.kind is EdgeKind.CONDITION_TAKEN) != zero:
                        return False, (), "known entry value makes the opposite branch outcome necessary"
                if instruction.operation is Operation.JNZ:
                    nonzero = registers[register] != 0
                    if (step.kind is EdgeKind.CONDITION_TAKEN) != nonzero:
                        return False, (), "known entry value makes the opposite branch outcome necessary"
        if instruction.branch_kind is BranchKind.CALL:
            forced = self._forced_call_result(instruction, step, conditions)
            if step.kind is EdgeKind.CALL_RETURN and forced is False:
                return False, (), "call result constraint says the callee does not return"
            if forced is not None:
                matching = self._matching_call_constraints(instruction, conditions)
                relationship = next(
                    (
                        item
                        for item in self.call_relationships
                        if item.call_instruction_id == instruction.instruction_id
                    ),
                    None,
                )
                if (
                    relationship is not None
                    and relationship.callee_may_return is not None
                    and relationship.callee_may_return == forced
                ):
                    return True, (), "analysis already established this call outcome"
                return True, tuple(item.constraint_id for item in matching), "call result constraint selected this continuation"
        return True, (), ""

    def _forced_call_result(
        self,
        instruction: Instruction,
        step: PathStep,
        conditions: PathConstraints,
    ) -> bool | None:
        matching = self._matching_call_constraints(instruction, conditions)
        values = {item.may_return for item in matching}
        if len(values) == 1:
            return next(iter(values))
        if step.kind is EdgeKind.CALL_RETURN:
            relationship = next(
                (
                    item
                    for item in self.call_relationships
                    if item.call_instruction_id == instruction.instruction_id
                ),
                None,
            )
            if relationship is not None and relationship.callee_may_return is not None:
                return relationship.callee_may_return
        return None

    def _matching_call_constraints(
        self,
        instruction: Instruction,
        conditions: PathConstraints,
    ) -> tuple[CallResultConstraint, ...]:
        return tuple(
            item
            for item in conditions.call_results
            if (item.call_address is None or item.call_address == instruction.address)
            and (
                item.callee_address is None
                or item.callee_address == instruction.branch_target
            )
        )

    def _update_registers(
        self,
        instruction: Instruction,
        registers: dict[int, int],
        step: PathStep,
    ) -> dict[int, int]:
        updated = dict(registers)
        if instruction.operation is Operation.LDI and len(instruction.operands) >= 2:
            updated[instruction.operands[0].value] = instruction.operands[1].value
        elif instruction.operation is Operation.SUBI and len(instruction.operands) >= 2:
            register = instruction.operands[0].value
            updated[register] = updated.get(register, 0) - instruction.operands[1].value
        elif instruction.operation in {Operation.ADD, Operation.SUB} and len(instruction.operands) >= 2:
            dest = instruction.operands[0].value
            source = instruction.operands[1].value
            left = updated.get(dest, 0)
            right = updated.get(source, 0)
            updated[dest] = left + right if instruction.operation is Operation.ADD else left - right
        return updated

    def _condition_frontier(
        self,
        target: int,
        origins: tuple[int, ...],
        conditions: PathConstraints,
        known_missing_ids: tuple[str, ...],
    ) -> tuple[list[ExcludedRelation], list[MissingBehavior], list[UnfinishedScope]]:
        excluded: list[ExcludedRelation] = []
        missing: list[MissingBehavior] = []
        unfinished: list[UnfinishedScope] = []
        instructions_by_id = {item.instruction_id: item for item in self.instructions}
        records = self._address_records()
        relations = list(self.edges) + [
            self._candidate_as_edge(item, instructions_by_id)
            for item in self.indirect_target_candidates
        ]
        initial = {
            item.register: item.value
            for item in conditions.entry_values
            if item.entry_address in origins
        }
        for edge in relations:
            instruction = instructions_by_id.get(edge.source_instruction_id)
            if instruction is None or instruction.address not in records:
                continue
            allowed, used, reason = self._step_allowed(instruction, edge, initial, conditions)
            if not allowed:
                constraint = next(
                    (item for item in conditions.branch_choices if item.branch_address == instruction.address),
                    None,
                )
                ids = (constraint.constraint_id,) if constraint else tuple(
                    item.constraint_id for item in self._matching_call_constraints(instruction, conditions)
                )
                excluded.append(
                    ExcludedRelation(
                        relation_id=edge.edge_id,
                        source_address=instruction.address,
                        target_address=edge.target_address,
                        reason=reason,
                        violated_constraint_ids=ids,
                    )
                )
        if self.budget_exhausted:
            for item in self.pending_targets:
                unfinished.append(
                    UnfinishedScope(
                        scope_id=f"scope:pending:{item.address:04x}",
                        kind="pending_target",
                        address=item.address,
                        description="budget stopped before this queued or unmapped target was resolved",
                        evidence=item.evidence,
                    )
                )
        for edge in self.edges:
            instruction = instructions_by_id.get(edge.source_instruction_id)
            if instruction is None:
                continue
            if edge.kind is EdgeKind.CALL_RETURN and not edge.resolved:
                matching = self._matching_call_constraints(instruction, conditions)
                if not matching:
                    requirement_id = f"missing:call-return:{instruction.address:04x}"
                    if requirement_id not in known_missing_ids:
                        missing.append(
                            MissingBehavior(
                                requirement_id=requirement_id,
                                kind="procedure_return",
                                subject_address=instruction.branch_target or instruction.address,
                                description="state whether the callee returns before accepting the return continuation",
                                relation_id=edge.edge_id,
                            )
                        )
            if edge.kind is EdgeKind.INDIRECT and not edge.resolved:
                missing.append(
                    MissingBehavior(
                        requirement_id=f"missing:indirect:{instruction.address:04x}",
                        kind="indirect_target",
                        subject_address=instruction.address,
                        description="provide a known register value or exact jump-table behavior",
                        relation_id=edge.edge_id,
                    )
                )
        return excluded, missing, unfinished

    def _candidate_as_edge(
        self,
        candidate: IndirectTargetCandidate,
        instructions_by_id: dict[str, Instruction],
    ) -> CfgEdge:
        return CfgEdge(
            edge_id=candidate.candidate_id,
            source_instruction_id=candidate.source_instruction_id,
            kind=EdgeKind.INDIRECT,
            target_address=candidate.target_address,
            resolved=False,
            evidence=candidate.evidence,
        )

    def to_dict(self) -> dict:
        return {
            "entry_points": list(self.entry_points),
            "roots": list(self.roots),
            "instructions": [_instruction_to_dict(item) for item in self.instructions],
            "edges": [_edge_to_dict(item) for item in self.edges],
            "blocks": [asdict(item) for item in self.blocks],
            "target_statuses": {
                str(address): status.value for address, status in self.target_statuses.items()
            },
            "unresolved_indirect_jumps": list(self.unresolved_indirect_jumps),
            "indirect_target_candidates": [
                _candidate_to_dict(item) for item in self.indirect_target_candidates
            ],
            "procedures": [asdict(item) for item in self.procedures],
            "call_relationships": [
                _call_relationship_to_dict(item) for item in self.call_relationships
            ],
            "budget_exhausted": self.budget_exhausted,
            "pending_targets": [_pending_to_dict(item) for item in self.pending_targets],
            "revisions": [_revision_to_dict(item) for item in self.revisions],
            "call_basis": self.call_basis.to_dict() if self.call_basis is not None else None,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "AnalysisResult":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        instructions = tuple(_instruction_from_dict(item) for item in data["instructions"])
        edges = tuple(_edge_from_dict(item) for item in data["edges"])
        blocks = tuple(
            BasicBlock(
                block_id=item["block_id"],
                start_address=item["start_address"],
                end_address=item["end_address"],
                instruction_ids=tuple(item["instruction_ids"]),
                outgoing_edge_ids=tuple(item["outgoing_edge_ids"]),
                overlap=item["overlap"],
            )
            for item in data["blocks"]
        )
        return cls(
            entry_points=tuple(data["entry_points"]),
            instructions=instructions,
            edges=edges,
            blocks=blocks,
            target_statuses={
                int(address): TargetStatus(status)
                for address, status in data["target_statuses"].items()
            },
            unresolved_indirect_jumps=tuple(data["unresolved_indirect_jumps"]),
            roots=tuple(data["roots"]),
            indirect_target_candidates=tuple(
                _candidate_from_dict(item) for item in data.get("indirect_target_candidates", ())
            ),
            procedures=tuple(
                Procedure(
                    procedure_id=item["procedure_id"],
                    entry_address=item["entry_address"],
                    may_return=item["may_return"],
                )
                for item in data.get("procedures", ())
            ),
            call_relationships=tuple(
                _call_relationship_from_dict(item)
                for item in data.get("call_relationships", ())
            ),
            budget_exhausted=data.get("budget_exhausted", False),
            pending_targets=tuple(
                _pending_from_dict(item) for item in data.get("pending_targets", ())
            ),
            revisions=tuple(
                _revision_from_dict(item) for item in data.get("revisions", ())
            ),
            call_basis=(
                CallBehaviorBasis.from_dict(data["call_basis"])
                if data.get("call_basis")
                else None
            ),
        )


def _evidence_to_dict(evidence: EdgeEvidence) -> dict:
    return {"kind": evidence.kind.value, "detail": evidence.detail}


def _pending_to_dict(item: PendingTarget) -> dict:
    return {
        "target_id": item.target_id,
        "address": item.address,
        "status": item.status.value,
        "evidence": [_evidence_to_dict(evidence) for evidence in item.evidence],
    }


def _pending_from_dict(data: dict) -> PendingTarget:
    return PendingTarget(
        target_id=data["target_id"],
        address=data["address"],
        status=TargetStatus(data["status"]),
        evidence=tuple(_evidence_from_dict(item) for item in data.get("evidence", ())),
    )


def _revision_to_dict(item: AnalysisRevision) -> dict:
    return {
        "revision_id": item.revision_id,
        "kind": item.kind.value,
        "subject_address": item.subject_address,
        "changed_relationship_ids": list(item.changed_relationship_ids),
        "detail": item.detail,
    }


def _revision_from_dict(data: dict) -> AnalysisRevision:
    return AnalysisRevision(
        revision_id=data["revision_id"],
        kind=RevisionKind(data["kind"]),
        subject_address=data["subject_address"],
        changed_relationship_ids=tuple(data.get("changed_relationship_ids", ())),
        detail=data.get("detail", ""),
    )


def origin_to_id(origins: tuple[int, ...] | tuple[int], address: int) -> str:
    try:
        return f"{origins.index(address):03d}"
    except ValueError:
        return "000"


def _default_edge_evidence(kind: EdgeKind) -> EdgeEvidence:
    if kind is EdgeKind.INDIRECT:
        return EdgeEvidence(EvidenceKind.KNOWN_CONSTANT)
    if kind is EdgeKind.CALL:
        return EdgeEvidence(EvidenceKind.CALL_OPERAND)
    if kind is EdgeKind.TAIL_CALL:
        return EdgeEvidence(EvidenceKind.TAIL_CALL)
    if kind is EdgeKind.CALL_RETURN:
        return EdgeEvidence(EvidenceKind.CALL_RETURN_UNKNOWN)
    return EdgeEvidence(EvidenceKind.DIRECT_OPERAND)


def _instruction_to_dict(instruction: Instruction) -> dict:
    return {
        "instruction_id": instruction.instruction_id,
        "address": instruction.address,
        "operation": instruction.operation.value,
        "operands": [asdict(item) for item in instruction.operands],
        "encoding": list(instruction.encoding),
        "next_address": instruction.next_address,
        "branch_kind": instruction.branch_kind.value,
        "branch_target": instruction.branch_target,
        "truncated": instruction.truncated,
        "unknown_opcode": instruction.unknown_opcode,
        "source_fragment_id": instruction.source_fragment_id,
    }


def _evidence_from_dict(data: dict) -> EdgeEvidence:
    return EdgeEvidence(EvidenceKind(data["kind"]), data.get("detail", ""))


def _edge_to_dict(edge: CfgEdge) -> dict:
    return {
        "edge_id": edge.edge_id,
        "source_instruction_id": edge.source_instruction_id,
        "kind": edge.kind.value,
        "target_address": edge.target_address,
        "target_block_id": edge.target_block_id,
        "resolved": edge.resolved,
        "evidence": [_evidence_to_dict(item) for item in edge.evidence],
    }


def _edge_from_dict(data: dict) -> CfgEdge:
    return CfgEdge(
        edge_id=data["edge_id"],
        source_instruction_id=data["source_instruction_id"],
        kind=EdgeKind(data["kind"]),
        target_address=data["target_address"],
        target_block_id=data["target_block_id"],
        resolved=data["resolved"],
        evidence=tuple(_evidence_from_dict(item) for item in data.get("evidence", ())),
    )


def _candidate_to_dict(candidate: IndirectTargetCandidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "source_instruction_id": candidate.source_instruction_id,
        "target_address": candidate.target_address,
        "status": candidate.status.value,
        "evidence": [_evidence_to_dict(item) for item in candidate.evidence],
    }


def _candidate_from_dict(data: dict) -> IndirectTargetCandidate:
    return IndirectTargetCandidate(
        candidate_id=data["candidate_id"],
        source_instruction_id=data["source_instruction_id"],
        target_address=data["target_address"],
        status=CandidateStatus(data["status"]),
        evidence=tuple(_evidence_from_dict(item) for item in data.get("evidence", ())),
    )


def _operand_from_dict(data: dict) -> Operand:
    return Operand(OperandKind(data["kind"]), data["value"])


def _instruction_from_dict(data: dict) -> Instruction:
    return Instruction(
        instruction_id=data["instruction_id"],
        address=data["address"],
        operation=Operation(data["operation"]),
        operands=tuple(_operand_from_dict(item) for item in data["operands"]),
        encoding=bytes(data["encoding"]),
        next_address=data["next_address"],
        branch_kind=BranchKind(data["branch_kind"]),
        branch_target=data["branch_target"],
        truncated=data["truncated"],
        unknown_opcode=data["unknown_opcode"],
        source_fragment_id=data["source_fragment_id"],
    )


def _call_relationship_to_dict(relationship: CallRelationship) -> dict:
    return {
        "call_instruction_id": relationship.call_instruction_id,
        "callee_address": relationship.callee_address,
        "return_address": relationship.return_address,
        "tail_call": relationship.tail_call,
        "callee_may_return": relationship.callee_may_return,
        "evidence": [_evidence_to_dict(item) for item in relationship.evidence],
    }


def _call_relationship_from_dict(data: dict) -> CallRelationship:
    return CallRelationship(
        call_instruction_id=data["call_instruction_id"],
        callee_address=data["callee_address"],
        return_address=data["return_address"],
        tail_call=data["tail_call"],
        callee_may_return=data["callee_may_return"],
        evidence=tuple(_evidence_from_dict(item) for item in data.get("evidence", ())),
    )


@dataclass(slots=True)
class _MutableResult:
    roots: list[int] = field(default_factory=list)
    instructions: dict[str, Instruction] = field(default_factory=dict)
    address_records: dict[int, list[str]] = field(default_factory=dict)
    edges: dict[str, CfgEdge] = field(default_factory=dict)
    target_statuses: dict[int, TargetStatus] = field(default_factory=dict)
    unresolved_indirect: list[str] = field(default_factory=list)
    indirect_candidates: dict[str, IndirectTargetCandidate] = field(default_factory=dict)
    pending_evidence: dict[int, list[EdgeEvidence]] = field(default_factory=dict)


def _edge_id(source_id: str, kind: EdgeKind, target_address: int | None) -> str:
    if target_address is None:
        return f"edge:{source_id}:{kind.value}"
    return f"edge:{source_id}:{kind.value}:{target_address:04x}"


class ControlFlowAnalyzer:
    """Recover code by following roots and control transfers.

    Unmapped gaps are not scanned. A jump to a future address is queued and
    decoded when reached. Different roots can decode overlapping byte ranges;
    both instruction identities and the overlap state are retained.
    """

    def __init__(
        self,
        address_space: AddressSpace,
        id_prefix: str = "ana",
        *,
        known_indirect_targets: list[KnownIndirectTarget] | tuple[KnownIndirectTarget, ...] = (),
        jump_tables: list[JumpTableConstraint] | tuple[JumpTableConstraint, ...] = (),
        noreturn_addresses: set[int] | frozenset[int] | tuple[int, ...] = (),
        return_overrides: dict[int, bool] | tuple[tuple[int, bool], ...] = (),
        call_behavior: CallBehaviorBasis | None = None,
    ) -> None:
        self.address_space = address_space
        self.id_prefix = id_prefix
        self.known_indirect_targets = tuple(known_indirect_targets)
        self.jump_tables = tuple(jump_tables)
        self.noreturn_addresses = frozenset(noreturn_addresses)
        self.return_overrides = dict(return_overrides)
        self.call_behavior = call_behavior
        self._instruction_counter = 0

    def _effective_noreturn(self) -> frozenset[int]:
        extra = set()
        if self.call_behavior is not None:
            for statement in self.call_behavior.usable_statements():
                if statement.may_return is False and statement.callee_address is not None:
                    extra.add(statement.callee_address)
        return self.noreturn_addresses | frozenset(extra)

    def analyze(self, *entry_points: int, budget: AnalysisBudget | None = None) -> AnalysisResult:
        state = _MutableResult(roots=list(dict.fromkeys(entry_points)))
        return self._run(state, tuple(entry_points), budget=budget)

    def resume(
        self,
        previous: AnalysisResult,
        budget: AnalysisBudget | None = None,
    ) -> AnalysisResult:
        state = self._state_from_result(previous)
        result = self._run(
            state,
            previous.entry_points,
            budget=budget,
            previous=previous,
            start_worklist=tuple(
                item.address
                for item in previous.pending_targets
                if item.status is TargetStatus.QUEUED
            ),
        )
        if budget is not None and previous.budget_exhausted:
            known = {item.instruction_id for item in previous.instructions}
            revision = AnalysisRevision(
                revision_id=f"{self.id_prefix}.revision.{len(previous.revisions) + 1:03d}",
                kind=RevisionKind.EXPLORATION_BUDGET,
                subject_address=previous.entry_points[0] if previous.entry_points else 0,
                changed_relationship_ids=tuple(
                    item.instruction_id
                    for item in result.instructions
                    if item.instruction_id not in known
                ),
                detail=f"exploration budget increased by max_instructions={budget.max_instructions}",
            )
            result = replace(result, revisions=result.revisions + (revision,))
        return result

    def apply_call_behavior(
        self,
        previous: AnalysisResult,
        basis: CallBehaviorBasis,
        budget: AnalysisBudget | None = None,
    ) -> RevisionReport:
        revised = ControlFlowAnalyzer(
            self.address_space,
            self.id_prefix,
            known_indirect_targets=self.known_indirect_targets,
            jump_tables=self.jump_tables,
            noreturn_addresses=self.noreturn_addresses,
            return_overrides=tuple(self.return_overrides.items()),
            call_behavior=basis,
        )
        result = revised.analyze(*previous.entry_points, budget=budget)
        old_returns = {
            edge.edge_id: edge.resolved
            for edge in previous.edges
            if edge.kind is EdgeKind.CALL_RETURN
        }
        new_returns = {
            edge.edge_id: edge.resolved
            for edge in result.edges
            if edge.kind is EdgeKind.CALL_RETURN
        }
        old_relationships = {
            item.call_instruction_id: item.callee_may_return
            for item in previous.call_relationships
        }
        new_relationships = {
            item.call_instruction_id: item.callee_may_return
            for item in result.call_relationships
        }
        changed = tuple(
            dict.fromkeys(
                [
                    edge_id
                    for edge_id in set(old_returns) | set(new_returns)
                    if old_returns.get(edge_id) != new_returns.get(edge_id)
                ]
                + [
                    call_id
                    for call_id in set(old_relationships) | set(new_relationships)
                    if old_relationships.get(call_id) != new_relationships.get(call_id)
                ]
            )
        )
        revision = AnalysisRevision(
            revision_id=f"{self.id_prefix}.revision.{len(previous.revisions) + 1:03d}",
            kind=RevisionKind.CALL_BASIS,
            subject_address=previous.entry_points[0] if previous.entry_points else 0,
            changed_relationship_ids=changed,
            detail=f"call behavior basis {basis.ref} applied",
        )
        result = self._append_revision(previous, result, revision)
        return RevisionReport(previous, result, revision)

    def revise_procedure_return(
        self,
        previous: AnalysisResult,
        procedure_address: int,
        may_return: bool,
        budget: AnalysisBudget | None = None,
    ) -> RevisionReport:
        revised = ControlFlowAnalyzer(
            self.address_space,
            self.id_prefix,
            known_indirect_targets=self.known_indirect_targets,
            jump_tables=self.jump_tables,
            noreturn_addresses=self.noreturn_addresses,
            return_overrides=(
                tuple(self.return_overrides.items()) + ((procedure_address, may_return),)
            ),
            call_behavior=self.call_behavior,
        )
        result = revised.analyze(*previous.entry_points, budget=budget)
        changed = tuple(
            edge.edge_id
            for edge in previous.edges
            if edge.kind is EdgeKind.CALL_RETURN
            and edge.target_address is not None
            and self._instruction_targets_procedure(edge, procedure_address, previous)
        ) + tuple(
            edge.edge_id
            for edge in result.edges
            if edge.kind is EdgeKind.CALL_RETURN
            and edge.target_address is not None
            and self._instruction_targets_procedure(edge, procedure_address, result)
        )
        revision = AnalysisRevision(
            revision_id=f"{self.id_prefix}.revision.{len(previous.revisions) + 1:03d}",
            kind=RevisionKind.PROCEDURE_RETURN,
            subject_address=procedure_address,
            changed_relationship_ids=changed,
            detail=f"may_return={may_return}",
        )
        result = self._append_revision(previous, result, revision)
        return RevisionReport(previous, result, revision)

    def revise_jump_table(
        self,
        previous: AnalysisResult,
        table: JumpTableConstraint,
        budget: AnalysisBudget | None = None,
    ) -> RevisionReport:
        tables = tuple(item for item in self.jump_tables if item.table_address != table.table_address) + (table,)
        revised = ControlFlowAnalyzer(
            self.address_space,
            self.id_prefix,
            known_indirect_targets=self.known_indirect_targets,
            jump_tables=tables,
            noreturn_addresses=self.noreturn_addresses,
            return_overrides=tuple(self.return_overrides.items()),
            call_behavior=self.call_behavior,
        )
        result = revised.analyze(*previous.entry_points, budget=budget)
        changed = tuple(
            candidate.candidate_id
            for candidate in result.indirect_target_candidates
            if any(
                evidence.kind
                in {EvidenceKind.JUMP_TABLE, EvidenceKind.TRUNCATED_JUMP_TABLE}
                and f"table=0x{table.table_address:04x}" in evidence.detail
                for evidence in candidate.evidence
            )
        ) + tuple(
            candidate.candidate_id
            for candidate in previous.indirect_target_candidates
            if any(
                evidence.kind
                in {EvidenceKind.JUMP_TABLE, EvidenceKind.TRUNCATED_JUMP_TABLE}
                and f"table=0x{table.table_address:04x}" in evidence.detail
                for evidence in candidate.evidence
            )
        ) + tuple(
            edge.edge_id
            for edge in result.edges
            if any(
                evidence.kind is EvidenceKind.JUMP_TABLE
                and f"table=0x{table.table_address:04x}" in evidence.detail
                for evidence in edge.evidence
            )
        ) + tuple(
            edge.edge_id
            for edge in previous.edges
            if any(
                evidence.kind is EvidenceKind.JUMP_TABLE
                and f"table=0x{table.table_address:04x}" in evidence.detail
                for evidence in edge.evidence
            )
        )
        revision = AnalysisRevision(
            revision_id=f"{self.id_prefix}.revision.{len(previous.revisions) + 1:03d}",
            kind=RevisionKind.JUMP_TABLE,
            subject_address=table.table_address,
            changed_relationship_ids=changed,
            detail=f"entries={len(table.entries or ())}",
        )
        result = self._append_revision(previous, result, revision)
        return RevisionReport(previous, result, revision)

    def _run(
        self,
        state: _MutableResult,
        entry_points: tuple[int, ...],
        *,
        budget: AnalysisBudget | None,
        previous: AnalysisResult | None = None,
        start_worklist: tuple[int, ...] = (),
    ) -> AnalysisResult:
        worklist = list(dict.fromkeys(start_worklist or state.roots))
        queued = set(worklist)
        for address in worklist:
            self._classify_target(address, state)

        budget_exhausted = False
        while worklist:
            if (
                budget is not None
                and budget.max_instructions is not None
                and len(state.instructions) >= self._budget_base(previous, budget)
            ):
                budget_exhausted = True
                for address in worklist:
                    self._classify_target(address, state)
                break
            address = worklist.pop(0)
            if state.address_records.get(address):
                continue
            if self.address_space.mapping_at(address) is None:
                state.target_statuses[address] = TargetStatus.UNMAPPED
                continue
            instruction = self._decode_at(address, state)
            self._queue_successor(instruction.next_address, instruction, worklist, queued, state)
            self._add_control_edges(instruction, worklist, queued, state)

        procedures, calls = self._build_procedures(state, budget_exhausted)
        blocks = self._build_blocks(state)
        pending = tuple(
            PendingTarget(
                target_id=f"pending:{address:04x}",
                address=address,
                status=status,
                evidence=tuple(state.pending_evidence.get(address, ())),
            )
            for address, status in sorted(state.target_statuses.items())
            if status is not TargetStatus.RESOLVED
        )
        result = AnalysisResult(
            entry_points=entry_points,
            roots=tuple(state.roots),
            instructions=tuple(state.instructions.values()),
            edges=tuple(state.edges.values()),
            blocks=tuple(blocks),
            target_statuses=dict(state.target_statuses),
            unresolved_indirect_jumps=tuple(state.unresolved_indirect),
            indirect_target_candidates=tuple(state.indirect_candidates.values()),
            procedures=procedures,
            call_relationships=calls,
            budget_exhausted=budget_exhausted,
            pending_targets=pending,
            revisions=previous.revisions if previous is not None else (),
            call_basis=self.call_behavior,
        )
        return self._reconnect_pending_evidence(result)

    def _budget_base(self, previous: AnalysisResult | None, budget: AnalysisBudget) -> int:
        if previous is None or budget.max_instructions is None:
            return budget.max_instructions or 0
        return len(previous.instructions) + budget.max_instructions

    def _state_from_result(self, previous: AnalysisResult) -> _MutableResult:
        self._instruction_counter = max(
            (self._counter_from_id(item.instruction_id, 4) for item in previous.instructions),
            default=0,
        )
        state = _MutableResult(roots=list(previous.roots))
        state.instructions = {item.instruction_id: item for item in previous.instructions}
        state.edges = {item.edge_id: item for item in previous.edges}
        state.target_statuses = dict(previous.target_statuses)
        state.unresolved_indirect = list(previous.unresolved_indirect_jumps)
        state.indirect_candidates = {
            item.candidate_id: item for item in previous.indirect_target_candidates
        }
        for item in previous.instructions:
            state.address_records.setdefault(item.address, []).append(item.instruction_id)
        for item in previous.pending_targets:
            state.pending_evidence[item.address] = list(item.evidence)
        return state

    def _counter_from_id(self, value: str, suffix_length: int) -> int:
        try:
            return int(value.rsplit(".", 1)[-1])
        except ValueError:
            return 0

    def _instruction_targets_procedure(
        self,
        edge: CfgEdge,
        procedure_address: int,
        result: AnalysisResult,
    ) -> bool:
        instruction = result.instruction(edge.source_instruction_id)
        return instruction.branch_target == procedure_address

    def _append_revision(
        self,
        previous: AnalysisResult,
        result: AnalysisResult,
        revision: AnalysisRevision,
    ) -> AnalysisResult:
        return replace(result, revisions=previous.revisions + (revision,))

    def _reconnect_pending_evidence(self, result: AnalysisResult) -> AnalysisResult:
        by_address: dict[int, list[EdgeEvidence]] = {}
        for edge in result.edges:
            if edge.target_address is not None:
                by_address.setdefault(edge.target_address, []).extend(edge.evidence)
        for candidate in result.indirect_target_candidates:
            by_address.setdefault(candidate.target_address, []).extend(candidate.evidence)
        pending = tuple(
            replace(
                item,
                evidence=tuple(
                    dict.fromkeys(list(by_address.get(item.address, ())) + list(item.evidence))
                ),
            )
            for item in result.pending_targets
        )
        return replace(result, pending_targets=pending)

    def _new_id(self) -> str:
        self._instruction_counter += 1
        return f"{self.id_prefix}.insn.{self._instruction_counter:04d}"

    def _decode_at(self, address: int, state: _MutableResult) -> Instruction:
        read = self.address_space.read(address, _MAX_INSTRUCTION_BYTES)
        location = self.address_space.location_at(address)
        fragment_id = location.fragment_id if location is not None else None
        instruction = decode_instruction(address, read.data, self._new_id(), fragment_id)
        state.instructions[instruction.instruction_id] = instruction
        state.address_records.setdefault(address, []).append(instruction.instruction_id)
        state.target_statuses[address] = TargetStatus.RESOLVED
        return instruction

    def _classify_target(
        self,
        address: int,
        state: _MutableResult,
        evidence: EdgeEvidence | None = None,
    ) -> None:
        if address in state.target_statuses:
            if evidence is not None:
                state.pending_evidence.setdefault(address, [])
                if evidence not in state.pending_evidence[address]:
                    state.pending_evidence[address].append(evidence)
            return
        state.target_statuses[address] = (
            TargetStatus.UNMAPPED
            if self.address_space.mapping_at(address) is None
            else TargetStatus.QUEUED
        )
        if evidence is not None:
            state.pending_evidence.setdefault(address, []).append(evidence)

    def _queue_address(self, address: int, worklist: list[int], queued: set[int]) -> None:
        if address not in queued:
            queued.add(address)
            worklist.append(address)

    def _queue_successor(
        self,
        address: int,
        instruction: Instruction,
        worklist: list[int],
        queued: set[int],
        state: _MutableResult,
    ) -> None:
        if instruction.truncated or instruction.unknown_opcode is not None:
            return
        if instruction.branch_kind in {BranchKind.FALLTHROUGH, BranchKind.CONDITIONAL, BranchKind.CALL}:
            if (
                instruction.branch_kind is BranchKind.CALL
                and instruction.branch_target in self._effective_noreturn()
            ):
                return
            self._classify_target(address, state)
            if state.target_statuses[address] is not TargetStatus.UNMAPPED:
                self._queue_address(address, worklist, queued)

    def _add_edge(
        self,
        instruction: Instruction,
        kind: EdgeKind,
        target_address: int | None,
        worklist: list[int],
        queued: set[int],
        state: _MutableResult,
        resolved: bool = True,
        evidence: tuple[EdgeEvidence, ...] = (),
    ) -> None:
        if target_address is not None and self.address_space.mapping_at(target_address) is None:
            resolved = False
        if not evidence:
            evidence = (_default_edge_evidence(kind),)
        edge = CfgEdge(
            edge_id=_edge_id(instruction.instruction_id, kind, target_address),
            source_instruction_id=instruction.instruction_id,
            kind=kind,
            target_address=target_address,
            target_block_id=None,
            resolved=resolved,
            evidence=evidence,
        )
        state.edges[edge.edge_id] = edge
        if target_address is not None:
            self._classify_target(target_address, state, edge.evidence[0] if edge.evidence else None)
            if resolved and state.target_statuses[target_address] is not TargetStatus.UNMAPPED:
                self._queue_address(target_address, worklist, queued)

    def _add_control_edges(
        self,
        instruction: Instruction,
        worklist: list[int],
        queued: set[int],
        state: _MutableResult,
    ) -> None:
        if instruction.truncated or instruction.unknown_opcode is not None:
            return
        kind = instruction.branch_kind
        if kind is BranchKind.FALLTHROUGH:
            self._add_edge(instruction, EdgeKind.FALLTHROUGH, instruction.next_address, worklist, queued, state)
        elif kind is BranchKind.UNCONDITIONAL:
            self._add_edge(instruction, EdgeKind.DIRECT_BRANCH, instruction.branch_target, worklist, queued, state)
        elif kind is BranchKind.CALL:
            self._add_edge(instruction, EdgeKind.CALL, instruction.branch_target, worklist, queued, state)
            if instruction.branch_target not in self.noreturn_addresses:
                self._add_edge(
                    instruction,
                    EdgeKind.CALL_RETURN,
                    instruction.next_address,
                    worklist,
                    queued,
                    state,
                    resolved=False,
                    evidence=(
                        EdgeEvidence(
                            EvidenceKind.CALL_RETURN_UNKNOWN,
                            f"callee=0x{instruction.branch_target:04x}",
                        ),
                    ),
                )
        elif kind is BranchKind.CONDITIONAL:
            assert instruction.branch_target is not None
            self._add_edge(instruction, EdgeKind.CONDITION_TAKEN, instruction.branch_target, worklist, queued, state)
            self._add_edge(instruction, EdgeKind.CONDITION_NOT_TAKEN, instruction.next_address, worklist, queued, state)
        elif kind is BranchKind.INDIRECT:
            self._add_indirect_edges(instruction, worklist, queued, state)

    def _add_indirect_edges(
        self,
        instruction: Instruction,
        worklist: list[int],
        queued: set[int],
        state: _MutableResult,
    ) -> None:
        confirmed = False
        register = instruction.operands[0].value if instruction.operands else None
        for known in self.known_indirect_targets:
            if known.source_address != instruction.address or register != known.register:
                continue
            self._add_edge(
                instruction,
                EdgeKind.INDIRECT,
                known.target_address,
                worklist,
                queued,
                state,
                evidence=(EdgeEvidence(EvidenceKind.KNOWN_CONSTANT, f"r{register}"),),
            )
            confirmed = True

        for table in self.jump_tables:
            if not table.matches(instruction):
                continue
            entries, complete = self._read_jump_table(table)
            evidence_kind = (
                EvidenceKind.JUMP_TABLE if complete else EvidenceKind.TRUNCATED_JUMP_TABLE
            )
            for index, target in enumerate(entries):
                candidate = IndirectTargetCandidate(
                    candidate_id=(
                        f"cand:{instruction.instruction_id}:{table.table_address:04x}:{index}"
                    ),
                    source_instruction_id=instruction.instruction_id,
                    target_address=target,
                    status=CandidateStatus.CANDIDATE if complete else CandidateStatus.TRUNCATED,
                    evidence=(
                        EdgeEvidence(
                            evidence_kind,
                            f"table=0x{table.table_address:04x},index={index}",
                        ),
                    ),
                )
                if complete:
                    self._add_edge(
                        instruction,
                        EdgeKind.INDIRECT,
                        target,
                        worklist,
                        queued,
                        state,
                        evidence=(candidate.evidence[0],),
                    )
                if not complete:
                    state.indirect_candidates[candidate.candidate_id] = candidate
                    self._classify_target(
                        target,
                        state,
                        candidate.evidence[0],
                    )
            if complete and entries:
                confirmed = True

        if not confirmed:
            state.unresolved_indirect.append(instruction.instruction_id)
            self._add_edge(
                instruction,
                EdgeKind.INDIRECT,
                None,
                worklist,
                queued,
                state,
                resolved=False,
            )

    def _read_jump_table(self, table: JumpTableConstraint) -> tuple[list[int], bool]:
        if table.entries is not None:
            count = len(table.entries)
            if table.entry_count is not None and count != table.entry_count:
                return list(table.entries), False
            return list(table.entries), count <= table.max_entries
        count = table.entry_count if table.entry_count is not None else table.max_entries
        if count > table.max_entries:
            return [], False
        count = min(count, table.max_entries)
        read = self.address_space.read(table.table_address, count * 2)
        entries = [
            (read.data[index] << 8) | read.data[index + 1]
            for index in range(0, len(read.data) - 1, 2)
        ]
        if table.entry_count is not None:
            return entries, len(entries) == table.entry_count and read.complete
        if not read.complete or len(read.data) % 2 == 1 or len(entries) < table.max_entries:
            return entries, False
        return entries, False

    def _build_procedures(
        self,
        state: _MutableResult,
        budget_exhausted: bool,
    ) -> tuple[tuple[Procedure, ...], tuple[CallRelationship, ...]]:
        basis_overrides: dict[int, bool] = {}
        if self.call_behavior is not None:
            for statement in self.call_behavior.usable_statements():
                if statement.may_return is None:
                    continue
                callee = statement.callee_address
                if callee is None and statement.call_address is not None:
                    record_ids = state.address_records.get(statement.call_address, ())
                    if record_ids:
                        callee = state.instructions[record_ids[0]].branch_target
                if callee is not None:
                    basis_overrides[callee] = statement.may_return
        overrides = {**basis_overrides, **self.return_overrides}
        call_entries = {
            edge.target_address
            for edge in state.edges.values()
            if edge.kind is EdgeKind.CALL and edge.target_address is not None
        }
        entries = set(call_entries)
        for edge in state.edges.values():
            target = edge.target_address
            if edge.kind is EdgeKind.DIRECT_BRANCH and target is not None:
                records = state.address_records.get(target, ())
                if records and state.instructions[records[0]].operation is Operation.RET:
                    entries.add(target)
        entries.update(self.noreturn_addresses)
        entries.update(overrides)
        may_return: dict[int, bool | None] = {
            address: overrides.get(
                address,
                False if address in self.noreturn_addresses else None,
            )
            for address in entries
        }
        for address in self.noreturn_addresses:
            may_return[address] = False

        def compute(address: int, current: dict[int, bool | None]) -> bool | None:
            if address in overrides:
                return overrides[address]
            if address in self.noreturn_addresses:
                return False

            def reachable_return(current_address: int, stack: tuple[int, ...]) -> bool | None:
                instruction_ids = state.address_records.get(current_address, ())
                if not instruction_ids:
                    return None
                instruction = state.instructions[instruction_ids[0]]
                if not instruction.is_valid:
                    return None
                if instruction.operation is Operation.RET:
                    return True
                if instruction.operation is Operation.HALT:
                    return False
                if instruction.branch_kind is BranchKind.CALL:
                    target = instruction.branch_target
                    if target is None:
                        return None
                    if target in stack:
                        return reachable_return(instruction.next_address, stack)
                    callee_returns = current.get(target)
                    if callee_returns is None:
                        return None
                    if callee_returns is False:
                        return False
                    return reachable_return(instruction.next_address, stack)
                successors = self._successors_for_return_analysis(instruction, state)
                if not successors:
                    return None
                outcomes = []
                for successor in successors:
                    if successor in stack:
                        continue
                    outcomes.append(reachable_return(successor, stack))
                if any(item is True for item in outcomes):
                    return True
                if any(item is None for item in outcomes) or len(outcomes) != len(successors):
                    return None
                return False

            value = reachable_return(address, (address,))
            if value is None and budget_exhausted:
                return None
            return value

        for _ in range(max(1, len(entries) + 2)):
            updated = {address: compute(address, may_return) for address in entries}
            if updated == may_return:
                break
            may_return = updated
        else:
            may_return = {address: compute(address, may_return) for address in entries}

        tail_entries = set(call_entries) | set(self.noreturn_addresses)
        for edge in state.edges.values():
            target = edge.target_address
            if (
                edge.kind is EdgeKind.DIRECT_BRANCH
                and target is not None
                and target in entries
            ):
                tail_entries.add(target)

        procedures = tuple(
            Procedure(
                procedure_id=f"{self.id_prefix}.proc.{index:03d}",
                entry_address=address,
                may_return=may_return.get(address),
            )
            for index, address in enumerate(sorted(tail_entries), 1)
        )

        relationships: list[CallRelationship] = []
        for instruction in state.instructions.values():
            if instruction.branch_kind is BranchKind.CALL and instruction.branch_target is not None:
                target = instruction.branch_target
                returns = may_return.get(target)
                if returns:
                    return_id = _edge_id(
                        instruction.instruction_id, EdgeKind.CALL_RETURN, instruction.next_address
                    )
                    old_return = state.edges.get(return_id)
                    state.edges[return_id] = CfgEdge(
                        edge_id=return_id,
                        source_instruction_id=instruction.instruction_id,
                        kind=EdgeKind.CALL_RETURN,
                        target_address=instruction.next_address,
                        target_block_id=old_return.target_block_id if old_return else None,
                        resolved=True,
                        evidence=(EdgeEvidence(EvidenceKind.RETURN_INSTRUCTION, f"callee=0x{target:04x}"),),
                    )
                elif returns is False:
                    state.edges.pop(
                        _edge_id(instruction.instruction_id, EdgeKind.CALL_RETURN, instruction.next_address),
                        None,
                    )
                relationships.append(
                    CallRelationship(
                        call_instruction_id=instruction.instruction_id,
                        callee_address=target,
                        return_address=instruction.next_address,
                        tail_call=False,
                        callee_may_return=returns,
                        evidence=(EdgeEvidence(EvidenceKind.CALL_OPERAND),),
                    )
                )
            if (
                instruction.branch_kind is BranchKind.UNCONDITIONAL
                and instruction.branch_target in tail_entries
            ):
                target = instruction.branch_target
                relationships.append(
                    CallRelationship(
                        call_instruction_id=instruction.instruction_id,
                        callee_address=target,
                        return_address=None,
                        tail_call=True,
                        callee_may_return=may_return.get(target),
                        evidence=(EdgeEvidence(EvidenceKind.TAIL_CALL),),
                    )
                )
                old_id = _edge_id(instruction.instruction_id, EdgeKind.DIRECT_BRANCH, target)
                old = state.edges.pop(old_id, None)
                if old is not None:
                    state.edges[_edge_id(instruction.instruction_id, EdgeKind.TAIL_CALL, target)] = CfgEdge(
                        edge_id=_edge_id(instruction.instruction_id, EdgeKind.TAIL_CALL, target),
                        source_instruction_id=instruction.instruction_id,
                        kind=EdgeKind.TAIL_CALL,
                        target_address=target,
                        resolved=old.resolved,
                        evidence=(EdgeEvidence(EvidenceKind.TAIL_CALL),),
                    )

        return procedures, tuple(relationships)

    def _successors_for_return_analysis(
        self,
        instruction: Instruction,
        state: _MutableResult,
    ) -> tuple[int, ...]:
        if instruction.branch_kind is BranchKind.INDIRECT:
            return tuple(
                edge.target_address
                for edge in state.edges.values()
                if edge.source_instruction_id == instruction.instruction_id
                and edge.kind is EdgeKind.INDIRECT
                and edge.resolved
                and edge.target_address is not None
            )
        if instruction.branch_kind is BranchKind.CONDITIONAL:
            successors = [instruction.next_address]
            if instruction.branch_target is not None:
                successors.append(instruction.branch_target)
            return tuple(successors)
        if instruction.branch_kind is BranchKind.FALLTHROUGH:
            return (instruction.next_address,)
        if instruction.branch_kind is BranchKind.UNCONDITIONAL:
            return (instruction.branch_target,) if instruction.branch_target is not None else ()
        return ()

    def _build_blocks(self, state: _MutableResult) -> list[BasicBlock]:
        by_id = state.instructions
        leaders = self._find_leaders(state)
        raw_blocks: list[tuple[int, list[str], bool]] = []

        for start in sorted(leaders):
            for leader_id in state.address_records.get(start, ()):
                current = by_id[leader_id]
                ids: list[str] = []
                overlap = len(state.address_records.get(current.address, ())) > 1
                while True:
                    ids.append(current.instruction_id)
                    overlap = overlap or self._has_byte_overlap(current, state)
                    next_address = current.next_address
                    boundary = (
                        current.truncated
                        or current.unknown_opcode is not None
                        or current.terminates_block
                        or current.branch_kind is BranchKind.CONDITIONAL
                        or next_address in leaders
                        or len(state.address_records.get(next_address, ())) != 1
                    )
                    if boundary:
                        break
                    next_id = state.address_records.get(next_address, [None])[0]
                    if next_id is None or next_id not in by_id:
                        break
                    current = by_id[next_id]
                raw_blocks.append((start, ids, overlap))

        raw_blocks.sort(key=lambda item: (item[0], item[1][0]))
        blocks: list[BasicBlock] = []
        for index, (start, ids, overlap) in enumerate(raw_blocks, 1):
            last_id = ids[-1]
            blocks.append(
                BasicBlock(
                    block_id=f"{self.id_prefix}.block.{index:03d}",
                    start_address=start,
                    end_address=by_id[last_id].next_address,
                    instruction_ids=tuple(ids),
                    outgoing_edge_ids=tuple(
                        edge.edge_id
                        for edge in state.edges.values()
                        if edge.source_instruction_id == last_id
                    ),
                    overlap=overlap,
                )
            )
        return self._connect_edges(blocks, state)

    def _has_byte_overlap(self, instruction: Instruction, state: _MutableResult) -> bool:
        for other in state.instructions.values():
            if other.instruction_id == instruction.instruction_id:
                continue
            if instruction.address < other.address < instruction.next_address:
                return True
            if other.address < instruction.address < other.next_address:
                return True
            if (
                instruction.address == other.address
                and instruction.encoding != other.encoding
            ):
                return True
        return False

    def _find_leaders(self, state: _MutableResult) -> set[int]:
        leaders = set(state.roots)
        for instruction in state.instructions.values():
            if len(state.address_records.get(instruction.address, ())) > 1:
                leaders.add(instruction.address)
            if instruction.branch_target is not None:
                leaders.add(instruction.branch_target)
            if instruction.branch_kind in {
                BranchKind.CONDITIONAL,
                BranchKind.CALL,
                BranchKind.INDIRECT,
                BranchKind.RETURN,
            } or instruction.operation is Operation.HALT or instruction.truncated or instruction.unknown_opcode is not None:
                leaders.add(instruction.next_address)
        return {
            address
            for address in leaders
            if state.address_records.get(address)
        }

    def _connect_edges(self, blocks: list[BasicBlock], state: _MutableResult) -> list[BasicBlock]:
        block_by_instruction = {
            instruction_id: block.block_id
            for block in blocks
            for instruction_id in block.instruction_ids
        }
        target_blocks: dict[int, str] = {}
        for block in blocks:
            target_blocks.setdefault(block.start_address, block.block_id)

        connected: list[CfgEdge] = []
        for edge in state.edges.values():
            target_id = None
            if edge.target_address is not None:
                records = state.address_records.get(edge.target_address, ())
                if len(records) == 1:
                    target_id = block_by_instruction.get(records[0])
                elif records:
                    target_id = target_blocks.get(edge.target_address)
            connected.append(
                CfgEdge(
                    edge_id=edge.edge_id,
                    source_instruction_id=edge.source_instruction_id,
                    kind=edge.kind,
                    target_address=edge.target_address,
                    target_block_id=target_id,
                    resolved=edge.resolved,
                    evidence=edge.evidence,
                )
            )
        state.edges = {edge.edge_id: edge for edge in connected}
        return blocks
