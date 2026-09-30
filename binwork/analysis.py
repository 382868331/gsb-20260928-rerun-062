"""Recursive-disassembly and control-flow recovery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True, slots=True)
class EdgeEvidence:
    kind: EvidenceKind
    detail: str = ""


@dataclass(frozen=True, slots=True)
class IndirectTargetCandidate:
    candidate_id: str
    source_instruction_id: str
    target_address: int
    status: CandidateStatus
    evidence: tuple[EdgeEvidence, ...]


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
        )


def _evidence_to_dict(evidence: EdgeEvidence) -> dict:
    return {"kind": evidence.kind.value, "detail": evidence.detail}


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
    ) -> None:
        self.address_space = address_space
        self.id_prefix = id_prefix
        self.known_indirect_targets = tuple(known_indirect_targets)
        self.jump_tables = tuple(jump_tables)
        self.noreturn_addresses = frozenset(noreturn_addresses)
        self._instruction_counter = 0

    def analyze(self, *entry_points: int, budget: AnalysisBudget | None = None) -> AnalysisResult:
        state = _MutableResult(roots=list(dict.fromkeys(entry_points)))
        worklist = list(state.roots)
        queued: set[int] = set(worklist)
        for address in worklist:
            self._classify_target(address, state)

        budget_exhausted = False
        while worklist:
            if budget is not None and budget.max_instructions is not None and len(state.instructions) >= budget.max_instructions:
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
        return AnalysisResult(
            entry_points=tuple(entry_points),
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
                state.indirect_candidates[candidate.candidate_id] = candidate
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
        may_return = {address: address not in self.noreturn_addresses for address in entries}

        def compute(address: int, current: dict[int, bool]) -> bool | None:
            if address in self.noreturn_addresses:
                return False
            pending = [address]
            seen: set[int] = set()
            uncertain = budget_exhausted
            while pending:
                current_address = pending.pop()
                if current_address in seen:
                    continue
                seen.add(current_address)
                instruction_ids = state.address_records.get(current_address, ())
                if not instruction_ids:
                    return None
                instruction = state.instructions[instruction_ids[0]]
                if not instruction.is_valid:
                    uncertain = True
                    continue
                if instruction.operation is Operation.RET:
                    return True
                if instruction.operation is Operation.HALT:
                    continue
                if instruction.branch_kind is BranchKind.CALL:
                    target = instruction.branch_target
                    if target is None or target not in current:
                        uncertain = True
                    elif current[target]:
                        pending.append(instruction.next_address)
                    else:
                        continue
                if instruction.branch_kind is BranchKind.INDIRECT:
                    targets = [
                        edge.target_address
                        for edge in state.edges.values()
                        if edge.source_instruction_id == instruction.instruction_id
                        and edge.kind is EdgeKind.INDIRECT
                        and edge.resolved
                        and edge.target_address is not None
                    ]
                    if not targets:
                        uncertain = True
                    pending.extend(target for target in targets if target != address)
                    continue
                if instruction.branch_kind is BranchKind.UNCONDITIONAL:
                    target = instruction.branch_target
                    if target is not None and state.address_records.get(target):
                        pending.append(target)
                    else:
                        uncertain = True
                    continue
                if instruction.branch_kind is BranchKind.CONDITIONAL:
                    target = instruction.branch_target
                    if target is not None:
                        pending.append(target)
                    pending.append(instruction.next_address)
                else:
                    pending.append(instruction.next_address)
            return None if uncertain else False

        for _ in range(max(1, len(entries) + 2)):
            updated = {address: compute(address, may_return) for address in entries}
            normalized = {address: bool(value) if value is not None else False for address, value in updated.items()}
            if normalized == may_return:
                may_return = updated
                break
            may_return = normalized
        else:
            may_return = {address: compute(address, {key: bool(value) for key, value in may_return.items()}) for address in entries}

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
