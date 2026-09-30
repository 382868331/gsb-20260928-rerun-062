"""Recursive-disassembly and control-flow recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from .address import AddressSpace
from .isa import BranchKind, Instruction, Operation, decode_instruction

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
    CALL_RETURN = "call_return"
    TAIL_CALL = "tail_call"
    INDIRECT = "indirect"


class EdgeState(str, Enum):
    CONFIRMED = "confirmed"
    CANDIDATE = "candidate"
    UNRESOLVED = "unresolved"


class AnalysisState(str, Enum):
    COMPLETE = "complete"
    BUDGET_EXHAUSTED = "budget_exhausted"


class CandidateKind(str, Enum):
    KNOWN_CONSTANT = "known_constant"
    BRANCH_TABLE = "branch_table"


class EdgeBasis(str, Enum):
    STATIC_ENCODING = "static_encoding"
    FALLTHROUGH = "fallthrough"
    PROCEDURE_ENTRY = "procedure_entry"
    RETURN_ANALYSIS = "return_analysis"
    INDIRECT_UNRESOLVED = "indirect_unresolved"


@dataclass(frozen=True, slots=True)
class CfgEdge:
    edge_id: str
    source_instruction_id: str
    kind: EdgeKind
    target_address: int | None = None
    target_block_id: str | None = None
    resolved: bool = True
    state: EdgeState = EdgeState.CONFIRMED
    basis: EdgeBasis = EdgeBasis.STATIC_ENCODING
    evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IndirectCandidate:
    candidate_id: str
    source_instruction_id: str
    target_address: int
    kind: CandidateKind
    state: EdgeState = EdgeState.CANDIDATE
    evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JumpTable:
    """Manually identified 16-bit branch table for one indirect jump."""

    table_id: str
    source_address: int
    table_address: int
    entry_count: int
    entry_size: int = 2

    def __post_init__(self) -> None:
        if self.source_address < 0 or self.table_address < 0:
            raise ValueError("source and table addresses must be non-negative")
        if not self.table_id:
            raise ValueError("table_id must not be empty")
        if self.entry_count < 0 or self.entry_size != 2:
            raise ValueError("branch table must contain a non-negative number of u16 entries")


@dataclass(frozen=True, slots=True)
class AnalysisBudget:
    max_instructions: int | None = None

    def __post_init__(self) -> None:
        if self.max_instructions is not None and self.max_instructions < 0:
            raise ValueError("max_instructions must be non-negative")


@dataclass(frozen=True, slots=True)
class ProcedureInfo:
    procedure_id: str
    entry_address: int
    may_return: bool | None
    incoming_call_edge_ids: tuple[str, ...]
    incoming_tail_call_edge_ids: tuple[str, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)


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
    indirect_candidates: tuple[IndirectCandidate, ...] = ()
    procedures: tuple[ProcedureInfo, ...] = ()
    analysis_state: AnalysisState = AnalysisState.COMPLETE
    instructions_decoded: int = 0
    budget: AnalysisBudget | None = None

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


@dataclass(slots=True)
class _MutableResult:
    roots: list[int] = field(default_factory=list)
    instructions: dict[str, Instruction] = field(default_factory=dict)
    address_records: dict[int, list[str]] = field(default_factory=dict)
    edges: dict[str, CfgEdge] = field(default_factory=dict)
    target_statuses: dict[int, TargetStatus] = field(default_factory=dict)
    unresolved_indirect: list[str] = field(default_factory=list)
    indirect_candidates: dict[str, IndirectCandidate] = field(default_factory=dict)
    instructions_decoded: int = 0


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

    def __init__(self, address_space: AddressSpace, id_prefix: str = "ana") -> None:
        self.address_space = address_space
        self.id_prefix = id_prefix
        self._instruction_counter = 0

    def analyze(
        self,
        *entry_points: int,
        jump_tables: list[JumpTable] | tuple[JumpTable, ...] = (),
        known_constants: Mapping[int, int | tuple[int, ...] | list[int]] | None = None,
        budget: int | AnalysisBudget | None = None,
    ) -> AnalysisResult:
        state = _MutableResult(roots=list(dict.fromkeys(entry_points)))
        if isinstance(budget, int):
            budget = AnalysisBudget(budget)
        budget = budget or AnalysisBudget()
        tables = {table.source_address: table for table in jump_tables}
        if len(tables) != len(jump_tables):
            raise ValueError("more than one branch table is attached to an indirect jump")
        known_constants = known_constants or {}
        worklist = list(state.roots)
        queued: set[int] = set(worklist)
        for address in worklist:
            self._classify_target(address, state)

        analysis_state = AnalysisState.COMPLETE
        worklist, analysis_state = self._run_worklist(
            worklist,
            queued,
            state,
            tables,
            known_constants,
            budget,
            analysis_state,
            follow_call_returns=False,
        )

        procedures = self._analyze_procedures([], state)
        for edge in tuple(state.edges.values()):
            if edge.kind is EdgeKind.CALL and edge.target_address in procedures:
                may_return = procedures[edge.target_address].may_return
                if may_return is False:
                    placeholder_id = _edge_id(edge.source_instruction_id, EdgeKind.FALLTHROUGH, state.instructions[edge.source_instruction_id].next_address)
                    state.edges.pop(placeholder_id, None)
                else:
                    source = state.instructions[edge.source_instruction_id]
                    self._classify_target(source.next_address, state)
                    if state.target_statuses[source.next_address] is not TargetStatus.UNMAPPED:
                        self._queue_address(source.next_address, worklist, queued)

        worklist, analysis_state = self._run_worklist(
            worklist,
            queued,
            state,
            tables,
            known_constants,
            budget,
            analysis_state,
            follow_call_returns=True,
        )

        blocks = self._build_blocks(state)
        self._finalize_call_returns(procedures, state)
        blocks = self._build_blocks(state, rebuild=True)
        procedures = self._analyze_procedures(blocks, state, procedures)
        return AnalysisResult(
            entry_points=tuple(entry_points),
            roots=tuple(state.roots),
            instructions=tuple(state.instructions.values()),
            edges=tuple(state.edges.values()),
            blocks=tuple(blocks),
            target_statuses=dict(state.target_statuses),
            unresolved_indirect_jumps=tuple(state.unresolved_indirect),
            indirect_candidates=tuple(state.indirect_candidates.values()),
            procedures=tuple(procedures.values()),
            analysis_state=analysis_state,
            instructions_decoded=state.instructions_decoded,
            budget=budget,
        )

    def _new_id(self) -> str:
        self._instruction_counter += 1
        return f"{self.id_prefix}.insn.{self._instruction_counter:04d}"

    def _run_worklist(
        self,
        worklist: list[int],
        queued: set[int],
        state: _MutableResult,
        tables: Mapping[int, JumpTable],
        known_constants: Mapping[int, int | tuple[int, ...] | list[int]],
        budget: AnalysisBudget,
        analysis_state: AnalysisState,
        follow_call_returns: bool,
    ) -> tuple[list[int], AnalysisState]:
        while worklist:
            if budget.max_instructions is not None and state.instructions_decoded >= budget.max_instructions:
                return worklist, AnalysisState.BUDGET_EXHAUSTED
            address = worklist.pop(0)
            if state.address_records.get(address):
                continue
            if self.address_space.mapping_at(address) is None:
                state.target_statuses[address] = TargetStatus.UNMAPPED
                continue
            instruction = self._decode_at(address, state)
            self._queue_successor(instruction.next_address, instruction, worklist, queued, state)
            self._add_control_edges(
                instruction,
                worklist,
                queued,
                state,
                tables.get(address),
                _normalized_constants(known_constants.get(address)),
                follow_call_returns=follow_call_returns,
            )
        return worklist, analysis_state

    def _decode_at(self, address: int, state: _MutableResult) -> Instruction:
        read = self.address_space.read(address, _MAX_INSTRUCTION_BYTES)
        location = self.address_space.location_at(address)
        fragment_id = location.fragment_id if location is not None else None
        instruction = decode_instruction(address, read.data, self._new_id(), fragment_id)
        state.instructions[instruction.instruction_id] = instruction
        state.address_records.setdefault(address, []).append(instruction.instruction_id)
        state.target_statuses[address] = TargetStatus.RESOLVED
        state.instructions_decoded += 1
        return instruction

    def _classify_target(self, address: int, state: _MutableResult) -> None:
        if address in state.target_statuses:
            return
        state.target_statuses[address] = (
            TargetStatus.UNMAPPED
            if self.address_space.mapping_at(address) is None
            else TargetStatus.QUEUED
        )

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
        if instruction.branch_kind in {BranchKind.FALLTHROUGH, BranchKind.CONDITIONAL}:
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
        edge_state: EdgeState = EdgeState.CONFIRMED,
        basis: EdgeBasis = EdgeBasis.STATIC_ENCODING,
        evidence: Mapping[str, object] | None = None,
        queue_target: bool = True,
    ) -> None:
        mapped = target_address is None or self.address_space.mapping_at(target_address) is not None
        if target_address is not None and not mapped:
            resolved = False
        edge = CfgEdge(
            edge_id=_edge_id(instruction.instruction_id, kind, target_address),
            source_instruction_id=instruction.instruction_id,
            kind=kind,
            target_address=target_address,
            target_block_id=None,
            resolved=resolved,
            state=edge_state,
            basis=basis,
            evidence={"mapped_target": mapped, **dict(evidence or {})},
        )
        state.edges[edge.edge_id] = edge
        if target_address is not None:
            self._classify_target(target_address, state)
            if queue_target and resolved and state.target_statuses[target_address] is not TargetStatus.UNMAPPED:
                self._queue_address(target_address, worklist, queued)

    def _add_control_edges(
        self,
        instruction: Instruction,
        worklist: list[int],
        queued: set[int],
        state: _MutableResult,
        jump_table: JumpTable | None = None,
        known_targets: tuple[int, ...] = (),
        follow_call_returns: bool = True,
    ) -> None:
        if instruction.truncated or instruction.unknown_opcode is not None:
            return
        kind = instruction.branch_kind
        if kind is BranchKind.FALLTHROUGH:
            self._add_edge(
                instruction,
                EdgeKind.FALLTHROUGH,
                instruction.next_address,
                worklist,
                queued,
                state,
                basis=EdgeBasis.FALLTHROUGH,
                evidence={"static_next_address": instruction.next_address},
            )
        elif kind is BranchKind.UNCONDITIONAL:
            self._add_edge(
                instruction,
                EdgeKind.DIRECT_BRANCH,
                instruction.branch_target,
                worklist,
                queued,
                state,
                evidence={"static_branch_target": instruction.branch_target},
            )
        elif kind is BranchKind.CALL:
            self._add_edge(
                instruction,
                EdgeKind.CALL,
                instruction.branch_target,
                worklist,
                queued,
                state,
                basis=EdgeBasis.PROCEDURE_ENTRY,
                evidence={"static_call_target": instruction.branch_target},
            )
            self._add_edge(
                instruction,
                EdgeKind.FALLTHROUGH,
                instruction.next_address,
                worklist,
                queued,
                state,
                basis=EdgeBasis.FALLTHROUGH,
                evidence={"return_address": instruction.next_address, "pending_return_analysis": True},
                edge_state=EdgeState.CANDIDATE,
                queue_target=follow_call_returns,
            )
        elif kind is BranchKind.CONDITIONAL:
            assert instruction.branch_target is not None
            self._add_edge(
                instruction,
                EdgeKind.CONDITION_TAKEN,
                instruction.branch_target,
                worklist,
                queued,
                state,
                evidence={"static_branch_target": instruction.branch_target},
            )
            self._add_edge(
                instruction,
                EdgeKind.CONDITION_NOT_TAKEN,
                instruction.next_address,
                worklist,
                queued,
                state,
                basis=EdgeBasis.FALLTHROUGH,
                evidence={"static_next_address": instruction.next_address},
            )
        elif kind is BranchKind.INDIRECT:
            state.unresolved_indirect.append(instruction.instruction_id)
            self._known_constant_candidates(instruction, known_targets, state)
            table_evidence = self._branch_table_candidates(instruction, jump_table, state)
            unresolved_evidence = {"reason": "register value is not statically unique"}
            unresolved_evidence.update(table_evidence)
            self._add_edge(
                instruction,
                EdgeKind.INDIRECT,
                None,
                worklist,
                queued,
                state,
                resolved=False,
                edge_state=EdgeState.UNRESOLVED,
                basis=EdgeBasis.INDIRECT_UNRESOLVED,
                evidence=unresolved_evidence,
            )

    def _known_constant_candidates(
        self,
        instruction: Instruction,
        known_targets: tuple[int, ...],
        state: _MutableResult,
    ) -> None:
        for index, target in enumerate(known_targets):
            candidate_id = f"cand:{instruction.instruction_id}:known:{index}:{target:04x}"
            state.indirect_candidates[candidate_id] = IndirectCandidate(
                candidate_id=candidate_id,
                source_instruction_id=instruction.instruction_id,
                target_address=target,
                kind=CandidateKind.KNOWN_CONSTANT,
                state=EdgeState.CANDIDATE,
                evidence={
                    "source_address": instruction.address,
                    "constant": target,
                    "mapped": self.address_space.mapping_at(target) is not None,
                },
            )

    def _branch_table_candidates(
        self,
        instruction: Instruction,
        jump_table: JumpTable | None,
        state: _MutableResult,
    ) -> dict[str, object]:
        if jump_table is None:
            return {}
        entries: list[object] = []
        truncated = False
        for index in range(jump_table.entry_count):
            table_address = jump_table.table_address + index * jump_table.entry_size
            read = self.address_space.read(table_address, jump_table.entry_size)
            if not read.complete or len(read.data) != jump_table.entry_size:
                truncated = True
                entries.append(
                    {
                        "index": index,
                        "address": table_address,
                        "state": "truncated",
                        "bytes_read": len(read.data),
                    }
                )
                continue
            target = int.from_bytes(read.data, "big")
            mapped = self.address_space.mapping_at(target) is not None
            candidate_id = f"cand:{instruction.instruction_id}:{jump_table.table_id}:{index}"
            state.indirect_candidates[candidate_id] = IndirectCandidate(
                candidate_id=candidate_id,
                source_instruction_id=instruction.instruction_id,
                target_address=target,
                kind=CandidateKind.BRANCH_TABLE,
                state=EdgeState.CANDIDATE,
                evidence={
                    "table_id": jump_table.table_id,
                    "table_address": jump_table.table_address,
                    "entry_index": index,
                    "entry_address": table_address,
                    "mapped": mapped,
                },
            )
            entries.append(
                {
                    "index": index,
                    "address": table_address,
                    "target_address": target,
                    "state": "candidate",
                    "mapped": mapped,
                }
            )
        return {
            "branch_table": {
                "table_id": jump_table.table_id,
                "table_address": jump_table.table_address,
                "entry_count": jump_table.entry_count,
                "truncated": truncated,
                "entries": tuple(entries),
            }
        }

    def _build_blocks(self, state: _MutableResult, rebuild: bool = False) -> list[BasicBlock]:
        by_id = state.instructions
        leaders = self._find_leaders(state)
        raw_blocks: list[tuple[int, list[str], bool]] = []
        if rebuild:
            self._instruction_counter = 0

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
        for address_records in state.address_records.values():
            if len(address_records) > 1:
                for instruction_id in address_records:
                    leaders.add(state.instructions[instruction_id].address)
        for instruction in state.instructions.values():
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

    def _analyze_procedures(
        self,
        blocks: list[BasicBlock],
        state: _MutableResult,
        previous: dict[int, ProcedureInfo] | None = None,
    ) -> dict[int, ProcedureInfo]:
        entries = set(state.roots)
        for edge in state.edges.values():
            if edge.kind in {EdgeKind.CALL, EdgeKind.TAIL_CALL} and edge.target_address is not None:
                entries.add(edge.target_address)

        may_return = {
            address: previous[address].may_return if previous is not None and address in previous else True
            for address in entries
        }
        iterations = 0
        for iterations in range(1, len(entries) + 2):
            updated = dict(may_return)
            for entry in entries:
                updated[entry] = self._procedure_may_return(entry, entries, updated, state)
            if updated == may_return:
                break
            may_return = updated

        return {
            entry: ProcedureInfo(
                procedure_id=f"{self.id_prefix}.proc.{entry:04x}",
                entry_address=entry,
                may_return=may_return[entry],
                incoming_call_edge_ids=tuple(
                    edge.edge_id
                    for edge in state.edges.values()
                    if edge.kind is EdgeKind.CALL and edge.target_address == entry
                ),
                incoming_tail_call_edge_ids=tuple(
                    edge.edge_id
                    for edge in state.edges.values()
                    if edge.kind is EdgeKind.TAIL_CALL and edge.target_address == entry
                ),
                evidence={"fixed_point_iterations": iterations},
            )
            for entry in sorted(entries)
        }

    def _procedure_may_return(
        self,
        entry: int,
        entries: set[int],
        may_return: Mapping[int, bool | None],
        state: _MutableResult,
    ) -> bool | None:
        records = state.address_records.get(entry)
        if not records:
            return None

        pending: list[int] = [entry]
        seen: set[int] = set()
        result: bool | None = True
        saw_return = False
        while pending:
            address = pending.pop(0)
            if address in seen or address not in state.address_records:
                continue
            seen.add(address)
            instruction_id = state.address_records[address][0]
            instruction = state.instructions[instruction_id]
            if instruction.truncated or instruction.unknown_opcode is not None:
                result = _combine_may_return(result, None)
                continue
            kind = instruction.branch_kind
            if instruction.operation is Operation.HALT:
                result = _combine_may_return(result, False)
            elif kind is BranchKind.FALLTHROUGH:
                if instruction.operation is Operation.HALT:
                    result = _combine_may_return(result, False)
                else:
                    pending.append(instruction.next_address)
            elif kind is BranchKind.RETURN:
                result = _combine_may_return(result, True)
                saw_return = True
            elif kind is BranchKind.CONDITIONAL:
                pending.append(instruction.next_address)
                if instruction.branch_target is not None:
                    pending.append(instruction.branch_target)
            elif kind is BranchKind.CALL:
                callee = may_return.get(instruction.branch_target, None)
                result = _combine_may_return(result, callee)
                if callee is not False:
                    pending.append(instruction.next_address)
            elif kind is BranchKind.UNCONDITIONAL:
                target = instruction.branch_target
                if target in entries:
                    result = _combine_may_return(result, may_return.get(target))
                elif target is not None:
                    pending.append(target)
                else:
                    result = _combine_may_return(result, None)
            elif kind is BranchKind.INDIRECT:
                result = _combine_may_return(result, None)
        if result is True and not saw_return:
            result = False
        return result

    def _finalize_call_returns(self, procedures: Mapping[int, ProcedureInfo], state: _MutableResult) -> None:
        for edge in list(state.edges.values()):
            if edge.kind is EdgeKind.DIRECT_BRANCH and edge.target_address in procedures:
                state.edges[edge.edge_id] = _replace_edge(
                    edge,
                    kind=EdgeKind.TAIL_CALL,
                    basis=EdgeBasis.PROCEDURE_ENTRY,
                    evidence={
                        "static_target": edge.target_address,
                        "procedure_id": procedures[edge.target_address].procedure_id,
                    },
                )

        call_returns: dict[str, CfgEdge] = {}
        for edge in state.edges.values():
            if edge.kind is not EdgeKind.CALL or edge.target_address is None:
                continue
            source = state.instructions[edge.source_instruction_id]
            return_address = source.next_address
            return_id = _edge_id(source.instruction_id, EdgeKind.CALL_RETURN, return_address)
            may_return = procedures.get(edge.target_address).may_return if edge.target_address in procedures else None
            if may_return is False:
                continue
            state_state = EdgeState.CONFIRMED if may_return is True else EdgeState.CANDIDATE
            mapped = self.address_space.mapping_at(return_address) is not None
            call_returns[return_id] = CfgEdge(
                edge_id=return_id,
                source_instruction_id=source.instruction_id,
                kind=EdgeKind.CALL_RETURN,
                target_address=return_address,
                target_block_id=None,
                resolved=mapped,
                state=state_state,
                basis=EdgeBasis.RETURN_ANALYSIS,
                evidence={
                        "mapped_target": mapped,
                        "call_edge_id": edge.edge_id,
                        "callee_entry": edge.target_address,
                        "return_address": return_address,
                        "may_return": may_return,
                    },
            )

        state.edges = {
            edge.edge_id: edge
            for edge in state.edges.values()
            if not (
                edge.kind is EdgeKind.FALLTHROUGH
                and edge.evidence.get("pending_return_analysis") is True
            )
        }
        state.edges.update(call_returns)

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
                    state=edge.state,
                    basis=edge.basis,
                    evidence=dict(edge.evidence),
                )
            )
        state.edges = {edge.edge_id: edge for edge in connected}
        return blocks


def _combine_may_return(left: bool | None, right: bool | None) -> bool | None:
    if left is False or right is False:
        return False
    if left is None or right is None:
        return None
    return True


def _normalized_constants(value: int | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        values = (value,)
    else:
        values = tuple(value)
    if any(not 0 <= item <= 0xFFFF for item in values):
        raise ValueError("known indirect target must be a 16-bit address")
    return tuple(dict.fromkeys(values))


def _replace_edge(edge: CfgEdge, **changes: object) -> CfgEdge:
    fields = {
        "edge_id": edge.edge_id,
        "source_instruction_id": edge.source_instruction_id,
        "kind": edge.kind,
        "target_address": edge.target_address,
        "target_block_id": edge.target_block_id,
        "resolved": edge.resolved,
        "state": edge.state,
        "basis": edge.basis,
        "evidence": dict(edge.evidence),
    }
    fields.update(changes)
    return CfgEdge(**fields)
