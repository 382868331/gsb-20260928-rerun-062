"""Recursive-disassembly and control-flow recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum

from .address import AddressSpace
from .isa import BranchKind, Instruction, Operand, OperandKind, Operation, decode_instruction

_MAX_INSTRUCTION_BYTES = 4


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


class ReachabilityStatus(str, Enum):
    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"


class ConstraintKind(str, Enum):
    ENTRY_VALUE = "entry_value"
    BRANCH_TAKEN = "branch_taken"
    BRANCH_NOT_TAKEN = "branch_not_taken"
    CALL_RETURNS = "call_returns"
    CALL_NORETURN = "call_noreturn"


class MissingBehaviorKind(str, Enum):
    UNRESOLVED_INDIRECT_JUMP = "unresolved_indirect_jump"
    TRUNCATED_JUMP_TABLE = "truncated_jump_table"
    UNRESOLVED_CALL_RETURN = "unresolved_call_return"
    QUEUED_TARGET = "queued_target"


class RevisionKind(str, Enum):
    PROCEDURE_RETURN = "procedure_return"
    JUMP_TABLE = "jump_table"


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

    @property
    def evidence(self) -> tuple[EdgeEvidence, ...]:
        return tuple(item for step in self.steps for item in step.evidence)


@dataclass(frozen=True, slots=True)
class PathQueryResult:
    source_address: int | None
    target_address: int
    paths: tuple[ReachablePath, ...]
    query_id: str | None = None
    reachability: ReachabilityStatus = ReachabilityStatus.REACHABLE
    conflicts: tuple["ConstraintConflict", ...] = ()
    missing_behaviors: tuple["MissingBehavior", ...] = ()
    excluded_segments: tuple["ExcludedSegment", ...] = ()

    @property
    def status(self) -> PathStatus:
        if any(path.status is PathStatus.CONFIRMED for path in self.paths):
            return PathStatus.CONFIRMED
        if any(path.status is PathStatus.ASSUMED for path in self.paths):
            return PathStatus.ASSUMED
        return PathStatus.UNRESOLVED

    def to_dict(self) -> dict:
        return {
            "source_address": self.source_address,
            "target_address": self.target_address,
            "query_id": self.query_id,
            "reachability": self.reachability.value,
            "status": self.status.value,
            "paths": [_reachable_path_to_dict(item) for item in self.paths],
            "conflicts": [_conflict_to_dict(item) for item in self.conflicts],
            "missing_behaviors": [
                _missing_behavior_to_dict(item) for item in self.missing_behaviors
            ],
            "excluded_segments": [
                _excluded_segment_to_dict(item) for item in self.excluded_segments
            ],
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass(frozen=True, slots=True)
class EntryValueCondition:
    register: int
    value: int


@dataclass(frozen=True, slots=True)
class BranchChoice:
    branch_address: int
    outcome: ConstraintKind


@dataclass(frozen=True, slots=True)
class CallOutcome:
    call_address: int
    outcome: ConstraintKind


@dataclass(frozen=True, slots=True)
class QueryConstraints:
    """Input conditions attached to a reachability query.

    Conditions constrain one specific path search; analysis results and
    relationship identities are shared across queries and never rewritten.
    """

    label: str = ""
    entry_values: tuple[EntryValueCondition, ...] = ()
    branches: tuple[BranchChoice, ...] = ()
    calls: tuple[CallOutcome, ...] = ()

    def with_changes(
        self,
        *,
        entry_values: tuple[EntryValueCondition, ...] | None = None,
        branches: tuple[BranchChoice, ...] | None = None,
        calls: tuple[CallOutcome, ...] | None = None,
        label: str | None = None,
    ) -> "QueryConstraints":
        return replace(
            self,
            label=self.label if label is None else label,
            entry_values=self.entry_values if entry_values is None else entry_values,
            branches=self.branches if branches is None else branches,
            calls=self.calls if calls is None else calls,
        )

    def identity_inputs(self) -> tuple:
        return (
            tuple(sorted((item.register, item.value) for item in self.entry_values)),
            tuple(sorted((item.branch_address, item.outcome.value) for item in self.branches)),
            tuple(sorted((item.call_address, item.outcome.value) for item in self.calls)),
        )


@dataclass(frozen=True, slots=True)
class ConstraintConflict:
    kind: str
    constraint: ConstraintKind
    address: int | None
    register: int | None
    description: str
    basis: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MissingBehavior:
    kind: MissingBehaviorKind
    address: int | None
    description: str
    relation_id: str | None = None
    instruction_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExcludedSegment:
    source_address: int
    target_address: int | None
    relation_id: str
    kind: EdgeKind
    reason: str
    constraint: ConstraintKind | None = None
    evidence: tuple[EdgeEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryComparison:
    query_id_a: str
    query_id_b: str
    reachability_a: ReachabilityStatus
    reachability_b: ReachabilityStatus
    excluded_only_in_a: tuple[ExcludedSegment, ...]
    excluded_only_in_b: tuple[ExcludedSegment, ...]
    paths_only_in_a: tuple[str, ...]
    paths_only_in_b: tuple[str, ...]

    def explain(self) -> str:
        lines = [
            f"{self.query_id_a}: {self.reachability_a.value}",
            f"{self.query_id_b}: {self.reachability_b.value}",
        ]
        for segment in self.excluded_only_in_a:
            lines.append(
                f"- {self.query_id_a} excludes 0x{segment.source_address:04x}->"
                f"{(f'0x{segment.target_address:04x}' if segment.target_address is not None else '?')}: {segment.reason}"
            )
        for segment in self.excluded_only_in_b:
            lines.append(
                f"- {self.query_id_b} excludes 0x{segment.source_address:04x}->"
                f"{(f'0x{segment.target_address:04x}' if segment.target_address is not None else '?')}: {segment.reason}"
            )
        return "\n".join(lines)


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
        constraints: QueryConstraints | None = None,
        include_statuses: frozenset[PathStatus] | set[PathStatus] | tuple[PathStatus, ...] = (
            PathStatus.CONFIRMED,
            PathStatus.ASSUMED,
            PathStatus.UNRESOLVED,
        ),
        max_paths: int = 32,
    ) -> PathQueryResult:  # noqa: D401 - query API
        origins = (source_address,) if source_address is not None else self.entry_points
        paths: list[ReachablePath] = []
        wanted = frozenset(include_statuses)
        effective_source = source_address if source_address is not None else (origins[0] if origins else None)
        query_id = query_identity(effective_source, target_address, constraints)

        conflicts = self._constraint_conflicts(constraints)
        if conflicts:
            return PathQueryResult(
                source_address=source_address,
                target_address=target_address,
                paths=(),
                query_id=query_id,
                reachability=ReachabilityStatus.CONFLICTING,
                conflicts=tuple(conflicts),
            )

        excluded: dict[tuple[int, int | None, str], ExcludedSegment] = {}
        for origin in origins:
            if PathStatus.CONFIRMED in wanted:
                paths.extend(
                    self._confirmed_paths(
                        origin,
                        target_address,
                        max_paths - len(paths),
                        constraints=constraints,
                        excluded=excluded,
                    )
                )
            if PathStatus.ASSUMED in wanted:
                paths.extend(
                    self._assumed_paths(
                        origin,
                        target_address,
                        max_paths - len(paths),
                        constraints=constraints,
                        excluded=excluded,
                    )
                )
            if len(paths) >= max_paths:
                break
        if not paths and target_address in origins:
            paths.append(
                ReachablePath(
                    path_id=self._path_id(
                        query_id, origin_to_id(origins, target_address), target_address, 0
                    ),
                    source_address=target_address,
                    target_address=target_address,
                    status=PathStatus.CONFIRMED,
                    steps=(),
                )
            )

        missing: list[MissingBehavior] = []
        reachability = ReachabilityStatus.REACHABLE if paths else ReachabilityStatus.UNREACHABLE
        if paths:
            missing = self._missing_behaviors_for_paths(paths)
            if all(path.status is PathStatus.ASSUMED for path in paths) and missing:
                reachability = ReachabilityStatus.INSUFFICIENT
        else:
            open_frontier: dict[str, MissingBehavior] = {}
            for origin in origins:
                for item in self._open_frontier(
                    origin, constraints=constraints, excluded=excluded
                ):
                    open_frontier.setdefault(
                        f"{item.kind.value}:{item.address}:{item.relation_id}", item
                    )
            missing.extend(open_frontier.values())
            if missing:
                reachability = ReachabilityStatus.INSUFFICIENT
                if PathStatus.UNRESOLVED in wanted:
                    paths.append(
                        ReachablePath(
                            path_id=self._path_id(query_id, f"{effective_source or 0:04x}", target_address, 0)
                            if constraints is not None
                            else f"path:unresolved:{effective_source or 0:04x}:{target_address:04x}",
                            source_address=effective_source or 0,
                            target_address=target_address,
                            status=PathStatus.UNRESOLVED,
                            steps=(),
                            target_pending=any(
                                item.address == target_address for item in self.pending_targets
                            ),
                        )
                    )

        return PathQueryResult(
            source_address=source_address,
            target_address=target_address,
            paths=tuple(paths[:max_paths]),
            query_id=query_id,
            reachability=reachability,
            missing_behaviors=tuple(missing),
            excluded_segments=tuple(self._dedup_excluded(excluded).values()),
        )

    def compare_queries(
        self,
        target_address: int,
        constraints_a: QueryConstraints,
        constraints_b: QueryConstraints,
        *,
        source_address: int | None = None,
    ) -> QueryComparison:
        """Run two condition sets and report which constraints prune which edges."""
        result_a = self.paths_to(
            target_address,
            source_address=source_address,
            constraints=constraints_a,
        )
        result_b = self.paths_to(
            target_address,
            source_address=source_address,
            constraints=constraints_b,
        )
        excluded_a = {item.relation_id: item for item in result_a.excluded_segments}
        excluded_b = {item.relation_id: item for item in result_b.excluded_segments}
        ids_a = {path.path_id for path in result_a.paths}
        ids_b = {path.path_id for path in result_b.paths}
        return QueryComparison(
            query_id_a=result_a.query_id or "",
            query_id_b=result_b.query_id or "",
            reachability_a=result_a.reachability,
            reachability_b=result_b.reachability,
            excluded_only_in_a=tuple(
                item for key, item in excluded_a.items() if key not in excluded_b
            ),
            excluded_only_in_b=tuple(
                item for key, item in excluded_b.items() if key not in excluded_a
            ),
            paths_only_in_a=tuple(sorted(ids_a - ids_b)),
            paths_only_in_b=tuple(sorted(ids_b - ids_a)),
        )

    def _path_id(self, query_id: str, source_token: str, target: int, index: int) -> str:
        digest = query_id.split(":", 1)[1] if ":" in query_id else query_id
        return f"path:{digest}:{source_token}:{target:04x}:{index + 1:03d}"

    def _constraint_conflicts(
        self, constraints: QueryConstraints | None
    ) -> list[ConstraintConflict]:
        if constraints is None:
            return []
        conflicts: list[ConstraintConflict] = []
        entry_by_register: dict[int, int] = {}
        for item in constraints.entry_values:
            prior = entry_by_register.get(item.register)
            if prior is not None and prior != item.value:
                conflicts.append(
                    ConstraintConflict(
                        kind="duplicate_entry_value",
                        constraint=ConstraintKind.ENTRY_VALUE,
                        address=None,
                        register=item.register,
                        description=(
                            f"entry register r{item.register} constrained to {prior} and {item.value}"
                        ),
                        basis=(
                            f"entry_value(r{item.register}={prior})",
                            f"entry_value(r{item.register}={item.value})",
                        ),
                    )
                )
            entry_by_register[item.register] = item.value

        instructions_by_address = {
            item.address: item for item in self.instructions
        }
        branch_by_address: dict[int, ConstraintKind] = {}
        for item in constraints.branches:
            instruction = instructions_by_address.get(item.branch_address)
            if instruction is None or instruction.branch_kind is not BranchKind.CONDITIONAL:
                conflicts.append(
                    ConstraintConflict(
                        kind="unknown_branch_site",
                        constraint=item.outcome,
                        address=item.branch_address,
                        register=None,
                        description=(
                            f"no decoded conditional branch at 0x{item.branch_address:04x}"
                        ),
                        basis=(f"branch@{item.branch_address:04x}",),
                    )
                )
                continue
            prior = branch_by_address.get(item.branch_address)
            if prior is not None and prior is not item.outcome:
                conflicts.append(
                    ConstraintConflict(
                        kind="contradictory_branch_choice",
                        constraint=item.outcome,
                        address=item.branch_address,
                        register=None,
                        description=(
                            f"branch at 0x{item.branch_address:04x} constrained both taken and not taken"
                        ),
                        basis=(prior.value, item.outcome.value),
                    )
                )
            branch_by_address[item.branch_address] = item.outcome

        call_by_address: dict[int, ConstraintKind] = {}
        for item in constraints.calls:
            instruction = instructions_by_address.get(item.call_address)
            if (
                instruction is None
                or instruction.branch_kind
                not in {BranchKind.CALL, BranchKind.UNCONDITIONAL}
            ):
                conflicts.append(
                    ConstraintConflict(
                        kind="unknown_call_site",
                        constraint=item.outcome,
                        address=item.call_address,
                        register=None,
                        description=f"no decoded call/tail-call at 0x{item.call_address:04x}",
                        basis=(f"call@{item.call_address:04x}",),
                    )
                )
                continue
            prior = call_by_address.get(item.call_address)
            if prior is not None and prior is not item.outcome:
                conflicts.append(
                    ConstraintConflict(
                        kind="contradictory_call_outcome",
                        constraint=item.outcome,
                        address=item.call_address,
                        register=None,
                        description=(
                            f"call at 0x{item.call_address:04x} constrained both returns and noreturn"
                        ),
                        basis=(prior.value, item.outcome.value),
                    )
                )
            relationship = next(
                (
                    call
                    for call in self.call_relationships
                    if (
                        self.instruction(call.call_instruction_id).address == item.call_address
                        and not call.tail_call
                    )
                ),
                None,
            )
            if relationship is not None and relationship.callee_may_return is not None:
                expected = (
                    ConstraintKind.CALL_RETURNS
                    if relationship.callee_may_return
                    else ConstraintKind.CALL_NORETURN
                )
                if expected is not item.outcome:
                    conflicts.append(
                        ConstraintConflict(
                            kind="call_outcome_contradicts_analysis",
                            constraint=item.outcome,
                            address=item.call_address,
                            register=None,
                            description=(
                                f"call at 0x{item.call_address:04x} is confirmed to "
                                + ("return" if relationship.callee_may_return else "not return")
                            ),
                            basis=(
                                f"analysis:callee_0x{relationship.callee_address:04x}_may_return="
                                f"{relationship.callee_may_return}",
                                item.outcome.value,
                            ),
                        )
                    )
            call_by_address[item.call_address] = item.outcome
        return conflicts

    def _dedup_excluded(
        self, excluded: dict[tuple[int, int | None, str], ExcludedSegment]
    ) -> dict[str, ExcludedSegment]:
        result: dict[str, ExcludedSegment] = {}
        for segment in excluded.values():
            result.setdefault(segment.relation_id, segment)
        return result

    def _confirmed_paths(
        self,
        source: int,
        target: int,
        limit: int,
        *,
        constraints: QueryConstraints | None,
        excluded: dict[tuple[int, int | None, str], ExcludedSegment],
    ) -> list[ReachablePath]:
        return self._search_paths(
            source,
            target,
            limit,
            use_candidates=False,
            constraints=constraints,
            excluded=excluded,
        )

    def _assumed_paths(
        self,
        source: int,
        target: int,
        limit: int,
        *,
        constraints: QueryConstraints | None,
        excluded: dict[tuple[int, int | None, str], ExcludedSegment],
    ) -> list[ReachablePath]:
        return [
            path
            for path in self._search_paths(
                source,
                target,
                limit,
                use_candidates=True,
                constraints=constraints,
                excluded=excluded,
            )
            if path.status is PathStatus.ASSUMED
        ]

    def _search_paths(
        self,
        source: int,
        target: int,
        limit: int,
        *,
        use_candidates: bool,
        constraints: QueryConstraints | None = None,
        excluded: dict[tuple[int, int | None, str], ExcludedSegment] | None = None,
    ) -> list[ReachablePath]:
        if limit <= 0:
            return []
        if excluded is None:
            excluded = {}
        outgoing: dict[int, list[PathStep]] = {}
        instructions_by_id = {item.instruction_id: item for item in self.instructions}
        records = self._address_records()

        def add_step(step: PathStep) -> None:
            outgoing.setdefault(step.source_address, []).append(step)

        for edge in self.edges:
            if edge.target_address is None:
                continue
            if not edge.resolved and edge.kind is not EdgeKind.CALL_RETURN:
                continue
            instruction = instructions_by_id.get(edge.source_instruction_id)
            if instruction is None:
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
                    confirmed=edge.resolved,
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

        current_query_id = query_identity(source, target, constraints)

        def make_path_id(index: int) -> str:
            if constraints is None:
                return f"path:{source:04x}:{target:04x}:{index:03d}"
            digest = current_query_id.split(":", 1)[1]
            return f"path:{digest}:{source:04x}:{target:04x}:{index:03d}"

        branch_choices = {
            item.branch_address: item.outcome
            for item in (constraints.branches if constraints is not None else ())
        }
        call_outcomes = {
            item.call_address: item.outcome
            for item in (constraints.calls if constraints is not None else ())
        }

        def step_allowed(step: PathStep) -> tuple[bool, str, ConstraintKind | None]:
            instruction = instructions_by_id.get(step.source_instruction_id or "")
            if instruction is None:
                return True, "", None
            choice = branch_choices.get(instruction.address)
            if choice is not None:
                if choice is ConstraintKind.BRANCH_TAKEN and step.kind is not EdgeKind.CONDITION_TAKEN:
                    return False, f"branch at 0x{instruction.address:04x} constrained taken", choice
                if choice is ConstraintKind.BRANCH_NOT_TAKEN and step.kind is not EdgeKind.CONDITION_NOT_TAKEN:
                    return False, f"branch at 0x{instruction.address:04x} constrained not-taken", choice
            outcome = call_outcomes.get(instruction.address)
            if (
                outcome is ConstraintKind.CALL_NORETURN
                and step.kind is EdgeKind.CALL_RETURN
            ):
                return False, f"call at 0x{instruction.address:04x} constrained noreturn", outcome
            return True, "", None

        def record_excluded(step: PathStep, reason: str, constraint: ConstraintKind | None) -> None:
            key = (step.source_address, step.target_address, step.relation_id)
            excluded.setdefault(
                key,
                ExcludedSegment(
                    source_address=step.source_address,
                    target_address=step.target_address,
                    relation_id=step.relation_id,
                    kind=step.kind,
                    reason=reason,
                    constraint=constraint,
                    evidence=step.evidence,
                ),
            )

        def walk(
            address: int,
            steps: list[PathStep],
            relations: set[str],
            assumed: set[str],
            visiting: set[int],
        ) -> None:
            if len(results) >= limit:
                return
            if address == target and (steps or source == target):
                results.append(
                    ReachablePath(
                        path_id=make_path_id(len(results) + 1),
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
                    )
                )
                return
            if address in visiting:
                return
            visiting.add(address)
            for step in outgoing.get(address, ()):
                if step.relation_id in relations:
                    continue
                allowed, reason, constraint = step_allowed(step)
                if not allowed:
                    record_excluded(step, reason, constraint)
                    continue
                relations.add(step.relation_id)
                added = [
                    assumption_id
                    for assumption_id in step.assumption_ids
                    if assumption_id not in assumed
                ]
                assumed.update(added)
                steps.append(step)
                walk(step.target_address or address, steps, relations, assumed, visiting)
                steps.pop()
                for assumption_id in added:
                    assumed.discard(assumption_id)
                relations.discard(step.relation_id)
                if len(results) >= limit:
                    break
            visiting.discard(address)

        walk(source, [], set(), set(), set())
        return results

    def _open_frontier(
        self,
        source: int,
        *,
        constraints: QueryConstraints | None,
        excluded: dict[tuple[int, int | None, str], ExcludedSegment],
    ) -> list[MissingBehavior]:
        """Find unresolved continuations reachable from ``source`` under constraints.

        An open frontier proves that absence of a confirmed path is a lack of
        information rather than genuine unreachability. Recursion and loops
        terminate the traversal; budget stops therefore leave the remaining
        queued/unresolved range listed here.
        """
        instructions_by_id = {item.instruction_id: item for item in self.instructions}
        steps_by_source: dict[int, list[PathStep]] = {}
        for step in self._all_path_steps(instructions_by_id, use_candidates=True):
            steps_by_source.setdefault(step.source_address, []).append(step)

        branch_choices = {
            item.branch_address: item.outcome
            for item in (constraints.branches if constraints is not None else ())
        }
        call_outcomes = {
            item.call_address: item.outcome
            for item in (constraints.calls if constraints is not None else ())
        }
        frontier: list[MissingBehavior] = []
        seen_relations: set[str] = set()

        def note_excluded(step: PathStep, reason: str, constraint: ConstraintKind | None) -> None:
            excluded.setdefault(
                (step.source_address, step.target_address, step.relation_id),
                ExcludedSegment(
                    source_address=step.source_address,
                    target_address=step.target_address,
                    relation_id=step.relation_id,
                    kind=step.kind,
                    reason=reason,
                    constraint=constraint,
                    evidence=step.evidence,
                ),
            )

        def visit(address: int, stack: tuple[int, ...]) -> None:
            if address in stack:
                return
            next_stack = stack + (address,)
            for step in steps_by_source.get(address, ()):
                if step.relation_id in seen_relations:
                    continue
                instruction = instructions_by_id.get(step.source_instruction_id or "")
                choice = branch_choices.get(address) if instruction is not None else None
                if choice is not None:
                    if choice is ConstraintKind.BRANCH_TAKEN and step.kind is not EdgeKind.CONDITION_TAKEN:
                        note_excluded(step, f"branch at 0x{address:04x} constrained taken", choice)
                        continue
                    if choice is ConstraintKind.BRANCH_NOT_TAKEN and step.kind is not EdgeKind.CONDITION_NOT_TAKEN:
                        note_excluded(step, f"branch at 0x{address:04x} constrained not-taken", choice)
                        continue
                outcome = call_outcomes.get(address)
                if outcome is ConstraintKind.CALL_NORETURN and step.kind is EdgeKind.CALL_RETURN:
                    note_excluded(step, f"call at 0x{address:04x} constrained noreturn", outcome)
                    continue
                seen_relations.add(step.relation_id)
                missing = self._missing_behavior_for_step(step, instruction)
                if missing is not None:
                    frontier.append(missing)
                    continue
                if step.target_address is not None:
                    if step.kind is EdgeKind.CALL_RETURN and not step.confirmed:
                        frontier.append(
                            MissingBehavior(
                                kind=MissingBehaviorKind.UNRESOLVED_CALL_RETURN,
                                address=step.source_address,
                                description=(
                                    f"return behavior of callee from call at 0x{step.source_address:04x} is unresolved"
                                ),
                                relation_id=step.relation_id,
                                instruction_id=step.source_instruction_id,
                            )
                        )
                    visit(step.target_address, next_stack)

        visit(source, ())
        return frontier

    def _missing_behavior_for_step(
        self,
        step: PathStep,
        instruction: Instruction | None,
    ) -> MissingBehavior | None:
        if step.target_address is None:
            if step.kind is EdgeKind.INDIRECT:
                return MissingBehavior(
                    kind=MissingBehaviorKind.UNRESOLVED_INDIRECT_JUMP,
                    address=step.source_address,
                    description=(
                        f"indirect jump at 0x{step.source_address:04x} needs a known target or a complete jump table"
                    ),
                    relation_id=step.relation_id,
                    instruction_id=step.source_instruction_id,
                )
            return None
        if step.kind is EdgeKind.INDIRECT and not step.confirmed:
            return MissingBehavior(
                kind=MissingBehaviorKind.TRUNCATED_JUMP_TABLE,
                address=step.source_address,
                description=(
                    f"jump table for JMPR at 0x{step.source_address:04x} is partial; exact boundary and entry reaching 0x{step.target_address:04x} must be specified"
                ),
                relation_id=step.relation_id,
                instruction_id=step.source_instruction_id,
            )
        pending = next(
            (item for item in self.pending_targets if item.address == step.target_address),
            None,
        )
        if pending is not None and pending.status is TargetStatus.QUEUED:
            return MissingBehavior(
                kind=MissingBehaviorKind.QUEUED_TARGET,
                address=pending.address,
                description=(
                    f"0x{pending.address:04x} is still queued when the analysis budget stopped"
                ),
                relation_id=step.relation_id,
                instruction_id=step.source_instruction_id,
            )
        return None

    def _missing_behaviors_for_paths(
        self, paths: list[ReachablePath]
    ) -> list[MissingBehavior]:
        instructions_by_id = {item.instruction_id: item for item in self.instructions}
        missing: dict[str, MissingBehavior] = {}
        for path in paths:
            for step in path.steps:
                instruction = instructions_by_id.get(step.source_instruction_id or "")
                item = self._missing_behavior_for_step(step, instruction)
                if item is not None:
                    missing[f"{item.kind.value}:{item.address}:{item.relation_id}"] = item
                if step.kind is EdgeKind.CALL_RETURN and not step.confirmed:
                    key = f"{MissingBehaviorKind.UNRESOLVED_CALL_RETURN.value}:{step.source_address}:{step.relation_id}"
                    missing.setdefault(
                        key,
                        MissingBehavior(
                            kind=MissingBehaviorKind.UNRESOLVED_CALL_RETURN,
                            address=step.source_address,
                            description=(
                                f"return behavior of callee from call at 0x{step.source_address:04x} is unresolved"
                            ),
                            relation_id=step.relation_id,
                            instruction_id=step.source_instruction_id,
                        ),
                    )
        return list(missing.values())

    def _all_path_steps(
        self,
        instructions_by_id: dict[str, Instruction],
        *,
        use_candidates: bool,
    ) -> list[PathStep]:
        records = self._address_records()
        steps: list[PathStep] = []
        for edge in self.edges:
            if not edge.resolved and edge.kind is not EdgeKind.CALL_RETURN:
                continue
            instruction = instructions_by_id.get(edge.source_instruction_id)
            if instruction is None:
                continue
            target_records = records.get(edge.target_address, ()) if edge.target_address is not None else ()
            steps.append(
                PathStep(
                    relation_id=edge.edge_id,
                    source_instruction_id=edge.source_instruction_id,
                    source_address=instruction.address,
                    target_instruction_id=target_records[0] if target_records else None,
                    target_address=edge.target_address,
                    kind=edge.kind,
                    confirmed=edge.resolved,
                    evidence=edge.evidence,
                )
            )
        if use_candidates:
            for candidate in self.indirect_target_candidates:
                instruction = instructions_by_id.get(candidate.source_instruction_id)
                if instruction is None:
                    continue
                target_records = records.get(candidate.target_address, ())
                steps.append(
                    PathStep(
                        relation_id=candidate.candidate_id,
                        source_instruction_id=candidate.source_instruction_id,
                        source_address=instruction.address,
                        target_instruction_id=target_records[0] if target_records else None,
                        target_address=candidate.target_address,
                        kind=EdgeKind.INDIRECT,
                        confirmed=False,
                        assumption_ids=(f"assume:{candidate.candidate_id}",),
                        evidence=candidate.evidence,
                    )
                )
        return steps

    def _address_records(self) -> dict[int, tuple[str, ...]]:
        records: dict[int, list[str]] = {}
        for instruction in self.instructions:
            records.setdefault(instruction.address, []).append(instruction.instruction_id)
        return {address: tuple(ids) for address, ids in records.items()}

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
        )


def _evidence_to_dict(evidence: EdgeEvidence) -> dict:
    return {"kind": evidence.kind.value, "detail": evidence.detail}


def _reachable_path_to_dict(path: ReachablePath) -> dict:
    return {
        "path_id": path.path_id,
        "source_address": path.source_address,
        "target_address": path.target_address,
        "status": path.status.value,
        "steps": [asdict(step) for step in path.steps],
        "assumption_ids": list(path.assumption_ids),
        "target_pending": path.target_pending,
    }


def _conflict_to_dict(item: ConstraintConflict) -> dict:
    return {
        "kind": item.kind,
        "constraint": item.constraint.value,
        "address": item.address,
        "register": item.register,
        "description": item.description,
        "basis": list(item.basis),
    }


def _missing_behavior_to_dict(item: MissingBehavior) -> dict:
    return {
        "kind": item.kind.value,
        "address": item.address,
        "description": item.description,
        "relation_id": item.relation_id,
        "instruction_id": item.instruction_id,
    }


def _excluded_segment_to_dict(item: ExcludedSegment) -> dict:
    return {
        "source_address": item.source_address,
        "target_address": item.target_address,
        "relation_id": item.relation_id,
        "kind": item.kind.value,
        "reason": item.reason,
        "constraint": item.constraint.value if item.constraint is not None else None,
        "evidence": [_evidence_to_dict(evidence) for evidence in item.evidence],
    }


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


def query_identity(
    source: int | None,
    target: int,
    constraints: QueryConstraints | None,
) -> str:
    """Stable identity for a (source, target, condition-set) query.

    Old proofs, candidates and query results stay independent because each
    condition set hashes into its own identity; changing one condition always
    produces a new query instead of rewriting the previous one.
    """
    payload = json.dumps(
        {
            "source": source,
            "target": target,
            "conditions": constraints.identity_inputs() if constraints is not None else None,
            "label": constraints.label if constraints is not None else "",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"query:{digest}"


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
    ) -> None:
        self.address_space = address_space
        self.id_prefix = id_prefix
        self.known_indirect_targets = tuple(known_indirect_targets)
        self.jump_tables = tuple(jump_tables)
        self.noreturn_addresses = frozenset(noreturn_addresses)
        self.return_overrides = dict(return_overrides)
        self._instruction_counter = 0

    def analyze(self, *entry_points: int, budget: AnalysisBudget | None = None) -> AnalysisResult:
        state = _MutableResult(roots=list(dict.fromkeys(entry_points)))
        return self._run(state, tuple(entry_points), budget=budget)

    def resume(
        self,
        previous: AnalysisResult,
        budget: AnalysisBudget | None = None,
    ) -> AnalysisResult:
        state = self._state_from_result(previous)
        return self._run(
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
                evidence=tuple(dict.fromkeys(by_address.get(item.address, ()) + list(item.evidence))),
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
                and instruction.branch_target in self.noreturn_addresses
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
        entries.update(self.return_overrides)
        may_return: dict[int, bool | None] = {
            address: self.return_overrides.get(address, address not in self.noreturn_addresses)
            for address in entries
        }
        for address in self.noreturn_addresses:
            may_return[address] = False

        def compute(address: int, current: dict[int, bool | None]) -> bool | None:
            if address in self.return_overrides:
                return self.return_overrides[address]
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
                outcomes = [
                    reachable_return(successor, stack)
                    for successor in successors
                    if successor not in stack or state.instructions.get(
                        state.address_records.get(successor, (None,))[0]
                    ) is not None
                ]
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
