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
    CALL_BEHAVIOR_SPEC = "call_behavior_spec"


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
    CALL_BEHAVIOR = "call_behavior"


class ContinuationKind(str, Enum):
    EXPLORATION = "exploration"
    ASSUMPTION_CHANGE = "assumption_change"


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
class SpecCondition:
    """One applicability condition on a call-behavior declaration.

    The declaration only applies to executions where ``register`` holds a
    value within ``[minimum, maximum]`` (inclusive) at the queried entry.
    A spec carrying conditions must never be generalized to the whole
    function: queries outside the region get no support from it.
    """

    register: int
    minimum: int
    maximum: int

    def holds(self, value: int) -> bool:
        return self.minimum <= value <= self.maximum

    def render(self) -> str:
        return f"r{self.register} in [{self.minimum}, {self.maximum}]"


class SpecApplicability(str, Enum):
    APPLICABLE = "applicable"
    INAPPLICABLE = "inapplicable"
    UNDETERMINED = "undetermined"


def _spec_applicability(
    spec: "CallBehaviorSpec",
    entry_values: dict[int, int],
) -> SpecApplicability:
    undetermined = False
    for condition in spec.conditions:
        if condition.register in entry_values:
            if not condition.holds(entry_values[condition.register]):
                return SpecApplicability.INAPPLICABLE
        else:
            undetermined = True
    return SpecApplicability.UNDETERMINED if undetermined else SpecApplicability.APPLICABLE


def _conditions_compatible(
    left: tuple[SpecCondition, ...],
    right: tuple[SpecCondition, ...],
) -> bool:
    right_by_register = {item.register: item for item in right}
    for condition in left:
        other = right_by_register.get(condition.register)
        if other is None:
            continue
        if condition.maximum < other.minimum or other.maximum < condition.minimum:
            return False
    return True


def _condition_intersection(
    left: tuple[SpecCondition, ...],
    right: tuple[SpecCondition, ...],
) -> tuple[SpecCondition, ...]:
    merged: dict[int, SpecCondition] = {}
    for condition in left + right:
        current = merged.get(condition.register)
        if current is None:
            merged[condition.register] = condition
        else:
            merged[condition.register] = SpecCondition(
                condition.register,
                max(current.minimum, condition.minimum),
                min(current.maximum, condition.maximum),
            )
    return tuple(merged[register] for register in sorted(merged))


def _conditions_within(
    new: tuple[SpecCondition, ...],
    old: tuple[SpecCondition, ...],
) -> bool:
    """True when every input region allowed by ``new`` is allowed by ``old``."""
    new_by_register = {item.register: item for item in new}
    for condition in old:
        updated = new_by_register.get(condition.register)
        if updated is None:
            return False
        if updated.minimum < condition.minimum or updated.maximum > condition.maximum:
            return False
    return True


@dataclass(frozen=True, slots=True)
class CallBehaviorSpec:
    """One versioned declaration of how an external call behaves.

    Users declare what a call may return, whether it returns at all, and
    which branch conditions apply; each revision produces a new version,
    and query conclusions cite the exact ``spec_id@version`` they used.
    """

    spec_id: str
    version: int
    callee_address: int | None = None
    call_address: int | None = None
    may_return: bool | None = None
    return_values: tuple[tuple[int, int], ...] = ()
    branch_choices: tuple[BranchChoiceConstraint, ...] = ()
    conditions: tuple[SpecCondition, ...] = ()
    description: str = ""

    @property
    def ref(self) -> str:
        return f"{self.spec_id}@v{self.version}"

    def matches_call(self, call_address: int, callee_address: int | None) -> bool:
        if self.call_address is not None and self.call_address != call_address:
            return False
        if self.callee_address is not None and self.callee_address != callee_address:
            return False
        return True


@dataclass(frozen=True, slots=True)
class BehaviorConflict:
    conflict_id: str
    spec_refs: tuple[str, ...]
    subject: str
    reason: str
    callee_addresses: tuple[int, ...] = ()
    overlap: tuple[SpecCondition, ...] = ()


@dataclass(frozen=True, slots=True)
class CallBehaviorRegistry:
    """Versioned store of call-behavior declarations.

    ``specs`` holds the active versions, ``history`` superseded or
    withdrawn versions, and ``withdrawn`` the retired spec ids. All
    mutating helpers return a new registry; old registries stay valid.
    """

    specs: tuple[CallBehaviorSpec, ...] = ()
    history: tuple[CallBehaviorSpec, ...] = ()
    withdrawn: tuple[str, ...] = ()

    def declare(self, spec: CallBehaviorSpec) -> "CallBehaviorRegistry":
        if any(item.spec_id == spec.spec_id for item in self.specs):
            raise ValueError(f"call behavior spec already declared: {spec.spec_id}")
        withdrawn = tuple(item for item in self.withdrawn if item != spec.spec_id)
        return replace(self, specs=self.specs + (spec,), withdrawn=withdrawn)

    def revise(self, spec_id: str, **changes) -> tuple["CallBehaviorRegistry", CallBehaviorSpec]:
        current = next((item for item in self.specs if item.spec_id == spec_id), None)
        if current is None:
            raise KeyError(f"unknown call behavior spec: {spec_id}")
        revised = replace(current, version=current.version + 1, **changes)
        specs = tuple(revised if item.spec_id == spec_id else item for item in self.specs)
        return replace(self, specs=specs, history=self.history + (current,)), revised

    def withdraw(self, spec_id: str) -> "CallBehaviorRegistry":
        current = next((item for item in self.specs if item.spec_id == spec_id), None)
        if current is None:
            raise KeyError(f"unknown call behavior spec: {spec_id}")
        return replace(
            self,
            specs=tuple(item for item in self.specs if item.spec_id != spec_id),
            history=self.history + (current,),
            withdrawn=self.withdrawn + (spec_id,),
        )

    def spec(self, spec_id: str) -> CallBehaviorSpec:
        for item in self.specs:
            if item.spec_id == spec_id:
                return item
        raise KeyError(spec_id)

    def conflicts(self) -> tuple[BehaviorConflict, ...]:
        conflicts: list[BehaviorConflict] = []
        active = [item for item in self.specs if item.may_return is not None]
        for left_index, left in enumerate(active):
            for right in active[left_index + 1:]:
                if left.may_return == right.may_return:
                    continue
                if not _specs_overlap(left, right):
                    continue
                subject = _spec_subject(left, right)
                callees = tuple(sorted({
                    address
                    for address in (left.callee_address, right.callee_address)
                    if address is not None
                }))
                overlap = _condition_intersection(left.conditions, right.conditions)
                region = (
                    " on " + ", ".join(item.render() for item in overlap)
                    if overlap
                    else ""
                )
                conflicts.append(
                    BehaviorConflict(
                        conflict_id=f"behavior-conflict:{left.ref}:{right.ref}",
                        spec_refs=(left.ref, right.ref),
                        subject=subject,
                        reason=(
                            "declared call behaviors disagree on whether the call returns"
                            + region
                        ),
                        callee_addresses=callees,
                        overlap=overlap,
                    )
                )
        return tuple(conflicts)


def _specs_overlap(left: CallBehaviorSpec, right: CallBehaviorSpec) -> bool:
    if (
        left.call_address is not None
        and right.call_address is not None
        and left.call_address != right.call_address
    ):
        return False
    if (
        left.callee_address is not None
        and right.callee_address is not None
        and left.callee_address != right.callee_address
    ):
        return False
    return _conditions_compatible(left.conditions, right.conditions)


def _spec_subject(left: CallBehaviorSpec, right: CallBehaviorSpec) -> str:
    call = next(
        (item.call_address for item in (left, right) if item.call_address is not None),
        None,
    )
    callee = next(
        (item.callee_address for item in (left, right) if item.callee_address is not None),
        None,
    )
    parts = []
    if call is not None:
        parts.append(f"call=0x{call:04x}")
    if callee is not None:
        parts.append(f"callee=0x{callee:04x}")
    return ",".join(parts) or "any-call"


@dataclass(frozen=True, slots=True)
class MissingBehavior:
    requirement_id: str
    kind: str
    subject_address: int
    description: str
    relation_id: str | None = None
    constraint_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _QueryBehaviorView:
    """Query-time resolution of conditional call-behavior specs.

    Specs decide conclusions only inside their declared condition region:
    ``forced_returns`` holds call continuations established by applicable
    specs, ``disabled_calls`` marks analysis-time spec decisions the query
    constraints contradict or leave undetermined, and ``missing`` carries
    the resulting behavior/condition requirements.
    """

    forced_returns: dict[str, tuple[bool, tuple[str, ...]]]
    disabled_calls: frozenset[str]
    missing: tuple[MissingBehavior, ...]
    conflicts: tuple[BehaviorConflict, ...]


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
    behavior_refs: tuple[str, ...] = ()


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
    behavior_refs: tuple[str, ...] = ()

    @property
    def evidence(self) -> tuple[EdgeEvidence, ...]:
        return tuple(item for step in self.steps for item in step.evidence)

    @property
    def proof_kind(self) -> str:
        return "conditional" if self.behavior_refs else "direct"


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
    behavior_refs: tuple[str, ...] = ()

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

    @property
    def proof_kind(self) -> str:
        """Classify the query conclusion: direct code proof, a conclusion
        depending on manual call-behavior conditions, or insufficient
        conditions to decide."""
        decidable = tuple(
            path
            for path in self.paths
            if path.status in (PathStatus.CONFIRMED, PathStatus.ASSUMED)
        )
        if any(path.behavior_refs for path in decidable):
            return "conditional"
        if decidable:
            return "direct"
        if self.missing_behaviors:
            return "insufficient"
        return "none"


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
class QueryImpact:
    """How one previously issued query fares under a behavior revision.

    Queries whose conclusions never cited the changed spec versions keep
    their old proof (``depends_on_changed_spec`` is False); dependent
    queries are re-derived and exposed as ``derived_query``.
    """

    query_id: str
    derived_query_id: str
    target_address: int
    previous_status: PathStatus
    status: PathStatus
    depends_on_changed_spec: bool
    previous_behavior_refs: tuple[str, ...] = ()
    behavior_refs: tuple[str, ...] = ()
    derived_query: PathQueryResult | None = None


@dataclass(frozen=True, slots=True)
class BehaviorImpactReport:
    previous_result: AnalysisResult
    result: AnalysisResult
    revision: AnalysisRevision
    changed_spec_refs: tuple[str, ...]
    affected_relation_ids: tuple[str, ...]
    affected_candidate_ids: tuple[str, ...]
    affected_scope_ids: tuple[str, ...]
    query_impacts: tuple[QueryImpact, ...] = ()


@dataclass(frozen=True, slots=True)
class BehaviorComparison:
    target_address: int
    left: PathQueryResult
    right: PathQueryResult
    left_spec_refs: tuple[str, ...]
    right_spec_refs: tuple[str, ...]
    differences: tuple[ConstraintDifference, ...]


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
class ContinuationRecord:
    """Why a resumed analysis exists: more exploration or new assumptions."""

    record_id: str
    kind: ContinuationKind
    detail: str
    spec_refs: tuple[str, ...] = ()
    budget_max_instructions: int | None = None


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
    behavior_refs: tuple[str, ...] = ()


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
    call_behavior_specs: tuple[CallBehaviorSpec, ...] = ()
    behavior_conflicts: tuple[BehaviorConflict, ...] = ()
    continuations: tuple[ContinuationRecord, ...] = ()

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
        view = self._query_behavior_view(conditions, origins)
        conflicts = self._constraint_conflicts(conditions, origins)
        query_id = self._query_id(target_address, source_address, conditions)
        conflicts.extend(
            ConstraintConflict(
                conflict_id=item.conflict_id,
                constraint_ids=item.spec_refs,
                subject=item.subject,
                reason=item.reason,
            )
            for item in view.conflicts
        )
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
                        view,
                    )
                    paths.extend(found)
                    missing.extend(scope)
                if PathStatus.ASSUMED in wanted:
                    found, scope = self._assumed_paths(
                        origin,
                        target_address,
                        max_paths - len(paths),
                        conditions,
                        view,
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
                view,
            )
            missing.extend(view.missing)
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
            behavior_refs=self._path_behavior_refs(tuple(paths[:max_paths])),
        )

    @staticmethod
    def _path_behavior_refs(paths: tuple[ReachablePath, ...]) -> tuple[str, ...]:
        return tuple(sorted({ref for path in paths for ref in path.behavior_refs}))

    def _query_behavior_view(
        self,
        conditions: PathConstraints,
        origins: tuple[int, ...],
    ) -> _QueryBehaviorView:
        entry_values: dict[int, int] = {}
        for item in conditions.entry_values:
            if item.entry_address in origins:
                entry_values[item.register] = item.value
        spec_by_ref = {spec.ref: spec for spec in self.call_behavior_specs}
        forced: dict[str, tuple[bool, tuple[str, ...]]] = {}
        disabled: set[str] = set()
        missing: list[MissingBehavior] = []
        for relationship in self.call_relationships:
            if relationship.tail_call:
                continue
            call_id = relationship.call_instruction_id
            call = self.instruction(call_id)
            if relationship.callee_may_return is not None and relationship.behavior_refs:
                statuses = tuple(
                    _spec_applicability(spec_by_ref[ref], entry_values)
                    for ref in relationship.behavior_refs
                    if ref in spec_by_ref
                )
                if any(item is SpecApplicability.INAPPLICABLE for item in statuses):
                    disabled.add(call_id)
                    missing.append(
                        MissingBehavior(
                            requirement_id=f"missing:call-return:{call.address:04x}",
                            kind="procedure_return",
                            subject_address=relationship.callee_address,
                            description=(
                                "query constraints fall outside the declared spec"
                                " conditions; state whether the callee returns for"
                                " this input region"
                            ),
                        )
                    )
                elif any(item is SpecApplicability.UNDETERMINED for item in statuses):
                    disabled.add(call_id)
                    needed = tuple(
                        condition.render()
                        for ref in relationship.behavior_refs
                        if ref in spec_by_ref
                        for condition in spec_by_ref[ref].conditions
                        if condition.register not in entry_values
                    )
                    missing.append(
                        MissingBehavior(
                            requirement_id=f"missing:spec-condition:{call.address:04x}",
                            kind="spec_condition",
                            subject_address=relationship.callee_address,
                            description=(
                                "establish "
                                + ", ".join(needed)
                                + " before relying on "
                                + "+".join(relationship.behavior_refs)
                            ),
                        )
                    )
                continue
            if relationship.callee_may_return is not None:
                continue
            candidates = [
                spec
                for spec in self.call_behavior_specs
                if spec.may_return is not None
                and spec.matches_call(call.address, relationship.callee_address)
            ]
            applicable = [
                spec
                for spec in candidates
                if _spec_applicability(spec, entry_values) is SpecApplicability.APPLICABLE
            ]
            if applicable:
                decisions = {spec.may_return for spec in applicable}
                if len(decisions) == 1:
                    forced[call_id] = (
                        next(iter(decisions)),
                        tuple(sorted(spec.ref for spec in applicable)),
                    )
                continue
            undetermined = [
                spec
                for spec in candidates
                if _spec_applicability(spec, entry_values) is SpecApplicability.UNDETERMINED
            ]
            if undetermined:
                needed = tuple(
                    condition.render()
                    for spec in undetermined
                    for condition in spec.conditions
                    if condition.register not in entry_values
                )
                missing.append(
                    MissingBehavior(
                        requirement_id=f"missing:spec-condition:{call.address:04x}",
                        kind="spec_condition",
                        subject_address=relationship.callee_address,
                        description=(
                            "establish "
                            + ", ".join(needed)
                            + " before relying on "
                            + "+".join(sorted(spec.ref for spec in undetermined))
                        ),
                    )
                )
        conflicts = tuple(
            conflict
            for conflict in self.behavior_conflicts
            if all(
                condition.register not in entry_values
                or condition.holds(entry_values[condition.register])
                for condition in conflict.overlap
            )
        )
        return _QueryBehaviorView(
            forced_returns=forced,
            disabled_calls=frozenset(disabled),
            missing=tuple(missing),
            conflicts=conflicts,
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
            _relation_differences(left_result, right_result),
        )

    def _confirmed_paths(
        self,
        source: int,
        target: int,
        limit: int,
        conditions: PathConstraints,
        view: _QueryBehaviorView,
    ) -> tuple[list[ReachablePath], list[MissingBehavior]]:
        return self._search_paths(source, target, limit, conditions=conditions, use_candidates=False, view=view)

    def _assumed_paths(
        self,
        source: int,
        target: int,
        limit: int,
        conditions: PathConstraints,
        view: _QueryBehaviorView,
    ) -> tuple[list[ReachablePath], list[MissingBehavior]]:
        found, missing = self._search_paths(source, target, limit, conditions=conditions, use_candidates=True, view=view)
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
        view: _QueryBehaviorView,
    ) -> tuple[list[ReachablePath], list[MissingBehavior]]:
        if limit <= 0:
            return []
        outgoing: dict[int, list[PathStep]] = {}
        instructions_by_id = {item.instruction_id: item for item in self.instructions}
        records = self._address_records()
        behavior_by_call = {
            item.call_instruction_id: item.behavior_refs
            for item in self.call_relationships
            if item.behavior_refs
        }
        behavior_by_call = {
            call_id: refs
            for call_id, refs in behavior_by_call.items()
            if call_id not in view.disabled_calls
        }
        for call_id, (_, refs) in view.forced_returns.items():
            behavior_by_call[call_id] = refs

        def add_step(step: PathStep) -> None:
            outgoing.setdefault(step.source_address, []).append(step)

        for edge in self.edges:
            if edge.target_address is None:
                continue
            instruction = instructions_by_id.get(edge.source_instruction_id)
            if instruction is None:
                continue
            if (
                edge.kind is EdgeKind.CALL_RETURN
                and edge.source_instruction_id in view.disabled_calls
            ):
                continue
            forced_return = self._forced_call_result(instruction, edge, conditions, view)
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
                        behavior_refs=behavior_by_call.get(edge.source_instruction_id, ()),
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
                    behavior_refs=behavior_by_call.get(edge.source_instruction_id, ()),
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
                        behavior_refs=tuple(sorted({ref for step in steps for ref in step.behavior_refs})),
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
                        view,
                    )
                    if not allowed:
                        continue
                    next_registers = self._update_registers(instruction, registers, step)
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
    ) -> str:
        if not conditions.ids():
            return "query:unconstrained"
        payload = json.dumps(
            {
                "source": source,
                "target": target,
                "entry_values": [asdict(item) for item in conditions.entry_values],
                "branch_choices": [asdict(item) for item in conditions.branch_choices],
                "call_results": [asdict(item) for item in conditions.call_results],
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
        view: _QueryBehaviorView,
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
            forced = self._forced_call_result(instruction, step, conditions, view)
            if step.kind is EdgeKind.CALL_RETURN and forced is False:
                return False, (), "call result constraint says the callee does not return"
            if forced is not None:
                matching = self._matching_call_constraints(instruction, conditions)
                return True, tuple(item.constraint_id for item in matching), "call result constraint selected this continuation"
        return True, (), ""

    def _forced_call_result(
        self,
        instruction: Instruction,
        step: PathStep,
        conditions: PathConstraints,
        view: _QueryBehaviorView,
    ) -> bool | None:
        view_forced = view.forced_returns.get(instruction.instruction_id)
        if view_forced is not None:
            return view_forced[0]
        if instruction.instruction_id in view.disabled_calls:
            return None
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
        view: _QueryBehaviorView,
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
            allowed, used, reason = self._step_allowed(instruction, edge, initial, conditions, view)
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
            if (
                edge.kind is EdgeKind.CALL_RETURN
                and edge.resolved
                and edge.source_instruction_id in view.disabled_calls
            ):
                excluded.append(
                    ExcludedRelation(
                        relation_id=edge.edge_id,
                        source_address=instruction.address,
                        target_address=edge.target_address,
                        reason="call behavior spec conditions not established for this query",
                    )
                )
            if edge.kind is EdgeKind.CALL_RETURN and not edge.resolved:
                matching = self._matching_call_constraints(instruction, conditions)
                if not matching and edge.source_instruction_id not in view.forced_returns:
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
            "call_behavior_specs": [_spec_to_dict(item) for item in self.call_behavior_specs],
            "behavior_conflicts": [_behavior_conflict_to_dict(item) for item in self.behavior_conflicts],
            "continuations": [_continuation_to_dict(item) for item in self.continuations],
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
            call_behavior_specs=tuple(
                _spec_from_dict(item) for item in data.get("call_behavior_specs", ())
            ),
            behavior_conflicts=tuple(
                _behavior_conflict_from_dict(item)
                for item in data.get("behavior_conflicts", ())
            ),
            continuations=tuple(
                _continuation_from_dict(item) for item in data.get("continuations", ())
            ),
        )


def _changed_spec_refs(
    old_specs: tuple[CallBehaviorSpec, ...],
    new_specs: tuple[CallBehaviorSpec, ...],
) -> tuple[str, ...]:
    old_by_id = {spec.spec_id: spec for spec in old_specs}
    new_by_id = {spec.spec_id: spec for spec in new_specs}
    changed: list[str] = []
    for spec_id, new_spec in new_by_id.items():
        old_spec = old_by_id.get(spec_id)
        if old_spec != new_spec:
            changed.append(new_spec.ref)
            if old_spec is not None:
                changed.append(old_spec.ref)
    for spec_id, old_spec in old_by_id.items():
        if spec_id not in new_by_id:
            changed.append(old_spec.ref)
    return tuple(dict.fromkeys(changed))


def _changed_spec_subjects(
    old_specs: tuple[CallBehaviorSpec, ...],
    new_specs: tuple[CallBehaviorSpec, ...],
) -> tuple[int, ...]:
    old_by_id = {spec.spec_id: spec for spec in old_specs}
    new_by_id = {spec.spec_id: spec for spec in new_specs}
    subjects: list[int] = []
    for spec_id in set(old_by_id) | set(new_by_id):
        old_spec = old_by_id.get(spec_id)
        new_spec = new_by_id.get(spec_id)
        if old_spec == new_spec:
            continue
        for spec in (old_spec, new_spec):
            if spec is None:
                continue
            for address in (spec.callee_address, spec.call_address):
                if address is not None and address not in subjects:
                    subjects.append(address)
    return tuple(subjects)


def _relation_differences(
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
    for result in (left_result, right_result):
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


def _spec_to_dict(spec: CallBehaviorSpec) -> dict:
    return {
        "spec_id": spec.spec_id,
        "version": spec.version,
        "callee_address": spec.callee_address,
        "call_address": spec.call_address,
        "may_return": spec.may_return,
        "return_values": [list(item) for item in spec.return_values],
        "branch_choices": [asdict(item) for item in spec.branch_choices],
        "conditions": [asdict(item) for item in spec.conditions],
        "description": spec.description,
    }


def _spec_from_dict(data: dict) -> CallBehaviorSpec:
    return CallBehaviorSpec(
        spec_id=data["spec_id"],
        version=data["version"],
        callee_address=data.get("callee_address"),
        call_address=data.get("call_address"),
        may_return=data.get("may_return"),
        return_values=tuple(tuple(item) for item in data.get("return_values", ())),
        branch_choices=tuple(
            BranchChoiceConstraint(
                constraint_id=item["constraint_id"],
                branch_address=item["branch_address"],
                taken=item["taken"],
                target_address=item.get("target_address"),
            )
            for item in data.get("branch_choices", ())
        ),
        conditions=tuple(
            SpecCondition(
                register=item["register"],
                minimum=item["minimum"],
                maximum=item["maximum"],
            )
            for item in data.get("conditions", ())
        ),
        description=data.get("description", ""),
    )


def _behavior_conflict_to_dict(conflict: BehaviorConflict) -> dict:
    return {
        "conflict_id": conflict.conflict_id,
        "spec_refs": list(conflict.spec_refs),
        "subject": conflict.subject,
        "reason": conflict.reason,
        "callee_addresses": list(conflict.callee_addresses),
        "overlap": [asdict(item) for item in conflict.overlap],
    }


def _behavior_conflict_from_dict(data: dict) -> BehaviorConflict:
    return BehaviorConflict(
        conflict_id=data["conflict_id"],
        spec_refs=tuple(data["spec_refs"]),
        subject=data["subject"],
        reason=data["reason"],
        callee_addresses=tuple(data.get("callee_addresses", ())),
        overlap=tuple(
            SpecCondition(
                register=item["register"],
                minimum=item["minimum"],
                maximum=item["maximum"],
            )
            for item in data.get("overlap", ())
        ),
    )


def _continuation_to_dict(record: ContinuationRecord) -> dict:
    return {
        "record_id": record.record_id,
        "kind": record.kind.value,
        "detail": record.detail,
        "spec_refs": list(record.spec_refs),
        "budget_max_instructions": record.budget_max_instructions,
    }


def _continuation_from_dict(data: dict) -> ContinuationRecord:
    return ContinuationRecord(
        record_id=data["record_id"],
        kind=ContinuationKind(data["kind"]),
        detail=data["detail"],
        spec_refs=tuple(data.get("spec_refs", ())),
        budget_max_instructions=data.get("budget_max_instructions"),
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
        "behavior_refs": list(relationship.behavior_refs),
    }


def _call_relationship_from_dict(data: dict) -> CallRelationship:
    return CallRelationship(
        call_instruction_id=data["call_instruction_id"],
        callee_address=data["callee_address"],
        return_address=data["return_address"],
        tail_call=data["tail_call"],
        callee_may_return=data["callee_may_return"],
        evidence=tuple(_evidence_from_dict(item) for item in data.get("evidence", ())),
        behavior_refs=tuple(data.get("behavior_refs", ())),
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
        call_behaviors: CallBehaviorRegistry | tuple[CallBehaviorSpec, ...] | list[CallBehaviorSpec] | None = None,
    ) -> None:
        self.address_space = address_space
        self.id_prefix = id_prefix
        self.known_indirect_targets = tuple(known_indirect_targets)
        self.jump_tables = tuple(jump_tables)
        self.noreturn_addresses = frozenset(noreturn_addresses)
        self.return_overrides = dict(return_overrides)
        self.call_behaviors = self._normalize_call_behaviors(call_behaviors)
        self._spec_returns = self._call_behavior_return_map()
        self._instruction_counter = 0

    @staticmethod
    def _normalize_call_behaviors(
        call_behaviors: CallBehaviorRegistry | tuple[CallBehaviorSpec, ...] | list[CallBehaviorSpec] | None,
    ) -> CallBehaviorRegistry:
        if call_behaviors is None:
            return CallBehaviorRegistry()
        if isinstance(call_behaviors, CallBehaviorRegistry):
            return call_behaviors
        registry = CallBehaviorRegistry()
        for spec in call_behaviors:
            registry = registry.declare(spec)
        return registry

    def _call_behavior_return_map(self) -> dict[int, tuple[bool, tuple[str, ...]] | None]:
        """Map callee address to (may_return, spec refs); None marks a"""
        """callee whose declarations disagree and must stay unknown."""
        conflicted = {
            address
            for conflict in self.call_behaviors.conflicts()
            for address in conflict.callee_addresses
        }
        mapping: dict[int, tuple[bool, tuple[str, ...]] | None] = {}
        for spec in self.call_behaviors.specs:
            if spec.callee_address is None or spec.may_return is None:
                continue
            if spec.callee_address in conflicted:
                continue
            current = mapping.get(spec.callee_address)
            if current is None and spec.callee_address in mapping:
                continue
            if current is None:
                mapping[spec.callee_address] = (spec.may_return, (spec.ref,))
            elif current[0] == spec.may_return:
                mapping[spec.callee_address] = (current[0], current[1] + (spec.ref,))
            else:
                mapping[spec.callee_address] = None
        return mapping

    def _with_registry(self, registry: CallBehaviorRegistry) -> "ControlFlowAnalyzer":
        return ControlFlowAnalyzer(
            self.address_space,
            self.id_prefix,
            known_indirect_targets=self.known_indirect_targets,
            jump_tables=self.jump_tables,
            noreturn_addresses=self.noreturn_addresses,
            return_overrides=tuple(self.return_overrides.items()),
            call_behaviors=registry,
        )

    def analyze(self, *entry_points: int, budget: AnalysisBudget | None = None) -> AnalysisResult:
        state = _MutableResult(roots=list(dict.fromkeys(entry_points)))
        return self._run(state, tuple(entry_points), budget=budget)

    def resume(
        self,
        previous: AnalysisResult,
        budget: AnalysisBudget | None = None,
        added_specs: tuple[CallBehaviorSpec, ...] | list[CallBehaviorSpec] = (),
    ) -> AnalysisResult:
        records: list[ContinuationRecord] = []
        if budget is not None and budget.max_instructions is not None:
            records.append(
                ContinuationRecord(
                    record_id=f"{self.id_prefix}.cont.{len(previous.continuations) + len(records) + 1:03d}",
                    kind=ContinuationKind.EXPLORATION,
                    detail=f"exploration budget increased by {budget.max_instructions} instructions",
                    budget_max_instructions=budget.max_instructions,
                )
            )
        analyzer = self
        if added_specs:
            registry = CallBehaviorRegistry(specs=tuple(previous.call_behavior_specs))
            refs: list[str] = []
            for spec in added_specs:
                if any(item.spec_id == spec.spec_id for item in registry.specs):
                    existing = registry.spec(spec.spec_id)
                    if not _conditions_within(spec.conditions, existing.conditions):
                        raise ValueError(
                            "continuation cannot widen existing spec conditions: "
                            + spec.spec_id
                        )
                    registry, revised = registry.revise(
                        spec.spec_id,
                        callee_address=spec.callee_address,
                        call_address=spec.call_address,
                        may_return=spec.may_return,
                        return_values=spec.return_values,
                        branch_choices=spec.branch_choices,
                        conditions=spec.conditions,
                        description=spec.description,
                    )
                    refs.append(revised.ref)
                else:
                    registry = registry.declare(spec)
                    refs.append(registry.spec(spec.spec_id).ref)
            records.append(
                ContinuationRecord(
                    record_id=f"{self.id_prefix}.cont.{len(previous.continuations) + len(records) + 1:03d}",
                    kind=ContinuationKind.ASSUMPTION_CHANGE,
                    detail="call behavior specs declared or revised",
                    spec_refs=tuple(refs),
                )
            )
            analyzer = self._with_registry(registry)
        state = analyzer._state_from_result(previous)
        result = analyzer._run(
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
        return replace(result, continuations=previous.continuations + tuple(records))

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
            call_behaviors=self.call_behaviors,
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
            call_behaviors=self.call_behaviors,
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

    def revise_call_behavior(
        self,
        previous: AnalysisResult,
        call_behaviors: CallBehaviorRegistry | tuple[CallBehaviorSpec, ...] | list[CallBehaviorSpec],
        queries: tuple[PathQueryResult, ...] | list[PathQueryResult] = (),
        budget: AnalysisBudget | None = None,
    ) -> BehaviorImpactReport:
        """Re-analyze under a revised registry and preview the impact.

        The old result and its query conclusions are left untouched; queries
        that never cited the changed spec versions stay valid, dependent
        queries are re-derived on the new result.
        """
        registry = self._normalize_call_behaviors(call_behaviors)
        revised = self._with_registry(registry)
        result = revised.analyze(*previous.entry_points, budget=budget)
        changed_refs = _changed_spec_refs(previous.call_behavior_specs, registry.specs)
        changed_subjects = _changed_spec_subjects(previous.call_behavior_specs, registry.specs)
        affected_relations = tuple(dict.fromkeys(
            self._affected_relations(previous, changed_refs, changed_subjects)
            + self._affected_relations(result, changed_refs, changed_subjects)
        ))
        affected_candidates = tuple(dict.fromkeys(
            self._affected_candidates(previous, changed_refs)
            + self._affected_candidates(result, changed_refs)
        ))
        affected_scopes = tuple(dict.fromkeys(
            self._affected_scopes(previous, changed_refs, changed_subjects)
            + self._affected_scopes(result, changed_refs, changed_subjects)
        ))
        subject = next(
            (
                address
                for address in changed_subjects
                if address is not None
            ),
            0,
        )
        revision = AnalysisRevision(
            revision_id=f"{self.id_prefix}.revision.{len(previous.revisions) + 1:03d}",
            kind=RevisionKind.CALL_BEHAVIOR,
            subject_address=subject,
            changed_relationship_ids=affected_relations,
            detail="changed=" + ",".join(changed_refs),
        )
        result = self._append_revision(previous, result, revision)
        impacts = tuple(
            self._query_impact(result, query, frozenset(changed_refs))
            for query in queries
        )
        return BehaviorImpactReport(
            previous_result=previous,
            result=result,
            revision=revision,
            changed_spec_refs=changed_refs,
            affected_relation_ids=affected_relations,
            affected_candidate_ids=affected_candidates,
            affected_scope_ids=affected_scopes,
            query_impacts=impacts,
        )

    def withdraw_call_behavior(
        self,
        previous: AnalysisResult,
        spec_id: str,
        queries: tuple[PathQueryResult, ...] | list[PathQueryResult] = (),
        budget: AnalysisBudget | None = None,
    ) -> BehaviorImpactReport:
        registry = CallBehaviorRegistry(specs=previous.call_behavior_specs).withdraw(spec_id)
        return self.revise_call_behavior(previous, registry, queries=queries, budget=budget)

    def compare_call_behaviors(
        self,
        previous: AnalysisResult,
        target_address: int,
        left: CallBehaviorRegistry | tuple[CallBehaviorSpec, ...] | list[CallBehaviorSpec],
        right: CallBehaviorRegistry | tuple[CallBehaviorSpec, ...] | list[CallBehaviorSpec],
        *,
        source_address: int | None = None,
    ) -> BehaviorComparison:
        left_result = self._with_registry(self._normalize_call_behaviors(left)).analyze(
            *previous.entry_points
        )
        right_result = self._with_registry(self._normalize_call_behaviors(right)).analyze(
            *previous.entry_points
        )
        left_query = left_result.paths_to(target_address, source_address=source_address)
        right_query = right_result.paths_to(target_address, source_address=source_address)
        return BehaviorComparison(
            target_address=target_address,
            left=left_query,
            right=right_query,
            left_spec_refs=tuple(spec.ref for spec in left_result.call_behavior_specs),
            right_spec_refs=tuple(spec.ref for spec in right_result.call_behavior_specs),
            differences=_relation_differences(left_query, right_query),
        )

    def _query_impact(
        self,
        result: AnalysisResult,
        query: PathQueryResult,
        changed_refs: frozenset[str],
    ) -> QueryImpact:
        derived = result.paths_to(
            query.target_address,
            source_address=query.source_address,
            constraints=query.constraints,
        )
        depends = bool(
            changed_refs & set(query.behavior_refs)
            or changed_refs & set(derived.behavior_refs)
        )
        return QueryImpact(
            query_id=query.query_id,
            derived_query_id=derived.query_id,
            target_address=query.target_address,
            previous_status=query.status,
            status=derived.status,
            depends_on_changed_spec=depends,
            previous_behavior_refs=query.behavior_refs,
            behavior_refs=derived.behavior_refs,
            derived_query=derived,
        )

    @staticmethod
    def _affected_relations(
        result: AnalysisResult,
        changed_refs: tuple[str, ...],
        changed_subjects: tuple[int, ...],
    ) -> tuple[str, ...]:
        subjects = set(changed_subjects)
        found: list[str] = []
        for edge in result.edges:
            if any(ref in evidence.detail for evidence in edge.evidence for ref in changed_refs):
                found.append(edge.edge_id)
                continue
            if edge.kind in {EdgeKind.CALL, EdgeKind.CALL_RETURN, EdgeKind.TAIL_CALL}:
                instruction = result.instruction(edge.source_instruction_id)
                if instruction.address in subjects or instruction.branch_target in subjects:
                    found.append(edge.edge_id)
        for relationship in result.call_relationships:
            if set(relationship.behavior_refs) & set(changed_refs) or relationship.callee_address in subjects:
                found.append(f"call:{relationship.call_instruction_id}")
        return tuple(found)

    @staticmethod
    def _affected_candidates(
        result: AnalysisResult,
        changed_refs: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            candidate.candidate_id
            for candidate in result.indirect_target_candidates
            if any(ref in evidence.detail for evidence in candidate.evidence for ref in changed_refs)
        )

    @staticmethod
    def _affected_scopes(
        result: AnalysisResult,
        changed_refs: tuple[str, ...],
        changed_subjects: tuple[int, ...],
    ) -> tuple[str, ...]:
        subjects = set(changed_subjects)
        return tuple(
            f"scope:pending:{item.address:04x}"
            for item in result.pending_targets
            if item.address in subjects
            or any(ref in evidence.detail for evidence in item.evidence for ref in changed_refs)
        )

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
            call_behavior_specs=self.call_behaviors.specs,
            behavior_conflicts=self.call_behaviors.conflicts(),
            continuations=previous.continuations if previous is not None else (),
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
                evidence=tuple(dict.fromkeys(list(by_address.get(item.address, ())) + list(item.evidence))),
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

    def _spec_declares_no_return(self, callee_address: int | None) -> bool:
        if callee_address is None:
            return False
        info = self._spec_returns.get(callee_address)
        return bool(info and info[0] is False)

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
            if instruction.branch_kind is BranchKind.CALL and (
                instruction.branch_target in self.noreturn_addresses
                or self._spec_declares_no_return(instruction.branch_target)
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
        entries.update(
            spec.callee_address
            for spec in self.call_behaviors.specs
            if spec.callee_address is not None
        )

        def initial_return(address: int) -> bool | None:
            if address in self.return_overrides:
                return self.return_overrides[address]
            if address in self.noreturn_addresses:
                return False
            info = self._spec_returns.get(address)
            if info is not None:
                return info[0]
            return None

        may_return: dict[int, bool | None] = {
            address: initial_return(address) for address in entries
        }
        for address in self.noreturn_addresses:
            may_return[address] = False

        spec_decisive: set[int] = set()

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
            spec_info = self._spec_returns.get(address)
            if value is None and spec_info is not None:
                spec_decisive.add(address)
                return spec_info[0]
            spec_decisive.discard(address)
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
                spec_info = self._spec_returns.get(target)
                spec_based = (
                    spec_info is not None
                    and target in spec_decisive
                    and target not in self.return_overrides
                    and target not in self.noreturn_addresses
                )
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
                        evidence=(
                            EdgeEvidence(
                                EvidenceKind.CALL_BEHAVIOR_SPEC,
                                f"spec={'+'.join(spec_info[1])},callee=0x{target:04x}",
                            )
                            if spec_based
                            else EdgeEvidence(EvidenceKind.RETURN_INSTRUCTION, f"callee=0x{target:04x}"),
                        ),
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
                        behavior_refs=spec_info[1] if spec_based else (),
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
                        behavior_refs=(
                            self._spec_returns[target][1]
                            if self._spec_returns.get(target) and target in spec_decisive
                            else ()
                        ),
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
