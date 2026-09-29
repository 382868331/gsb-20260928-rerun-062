"""Recursive-disassembly and control-flow recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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
    INDIRECT = "indirect"


@dataclass(frozen=True, slots=True)
class CfgEdge:
    edge_id: str
    source_instruction_id: str
    kind: EdgeKind
    target_address: int | None = None
    target_block_id: str | None = None
    resolved: bool = True


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

    def analyze(self, *entry_points: int) -> AnalysisResult:
        state = _MutableResult(roots=list(dict.fromkeys(entry_points)))
        worklist = list(state.roots)
        queued: set[int] = set(worklist)
        for address in worklist:
            self._classify_target(address, state)

        while worklist:
            address = worklist.pop(0)
            if state.address_records.get(address):
                continue
            if self.address_space.mapping_at(address) is None:
                state.target_statuses[address] = TargetStatus.UNMAPPED
                continue
            instruction = self._decode_at(address, state)
            self._queue_successor(instruction.next_address, instruction, worklist, queued, state)
            self._add_control_edges(instruction, worklist, queued, state)

        blocks = self._build_blocks(state)
        return AnalysisResult(
            entry_points=tuple(entry_points),
            roots=tuple(state.roots),
            instructions=tuple(state.instructions.values()),
            edges=tuple(state.edges.values()),
            blocks=tuple(blocks),
            target_statuses=dict(state.target_statuses),
            unresolved_indirect_jumps=tuple(state.unresolved_indirect),
        )

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
        if instruction.branch_kind in {BranchKind.FALLTHROUGH, BranchKind.CONDITIONAL, BranchKind.CALL}:
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
    ) -> None:
        if target_address is not None and self.address_space.mapping_at(target_address) is None:
            resolved = False
        edge = CfgEdge(
            edge_id=_edge_id(instruction.instruction_id, kind, target_address),
            source_instruction_id=instruction.instruction_id,
            kind=kind,
            target_address=target_address,
            target_block_id=None,
            resolved=resolved,
        )
        state.edges[edge.edge_id] = edge
        if target_address is not None:
            self._classify_target(target_address, state)
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
            self._add_edge(instruction, EdgeKind.FALLTHROUGH, instruction.next_address, worklist, queued, state)
        elif kind is BranchKind.CONDITIONAL:
            assert instruction.branch_target is not None
            self._add_edge(instruction, EdgeKind.CONDITION_TAKEN, instruction.branch_target, worklist, queued, state)
            self._add_edge(instruction, EdgeKind.CONDITION_NOT_TAKEN, instruction.next_address, worklist, queued, state)
        elif kind is BranchKind.INDIRECT:
            state.unresolved_indirect.append(instruction.instruction_id)
            self._add_edge(instruction, EdgeKind.INDIRECT, None, worklist, queued, state, resolved=False)

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
                )
            )
        state.edges = {edge.edge_id: edge for edge in connected}
        return blocks
