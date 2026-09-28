"""控制流恢复（递归下降式，而非线性猜测）。

从已知入口出发，把直接跳转目标放入工作队列：目标可能落入尚未分析的
区域，也可能未映射、指向截断/未知指令。所有不能静态确定的后继都登记为
``UnresolvedEdge``，保留明确原因；间接跳转与 RET 绝不猜测目标。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Deque, Dict, List, Optional, Set, Tuple

from .disasm import DisassemblyDB
from .isa import DecodeStatus, DecodedInstruction, FlowKind


class EdgeKind(str, Enum):
    FALLTHROUGH = "fallthrough"
    BRANCH_TAKEN = "branch_taken"
    CALL_EDGE = "call"


class UnresolvedReason(str, Enum):
    UNMAPPED_TARGET = "unmapped_target"      # 跳转到未映射地址
    BAD_TARGET = "bad_target"              # 目标处截断或是未知操作码
    INDIRECT_JUMP = "indirect_jump"        # JMPIND：目标来自寄存器
    DYNAMIC_RETURN = "dynamic_return"      # RET：返回地址动态确定
    DECODE_ERROR = "decode_error"          # 当前指令无法解码，后继未知


@dataclass(frozen=True)
class Edge:
    edge_id: int
    source: int
    target: int
    kind: EdgeKind

    def as_tuple(self) -> Tuple[int, int, EdgeKind]:
        return (self.source, self.target, self.kind)


@dataclass(frozen=True)
class UnresolvedEdge:
    source: int
    kind: UnresolvedReason
    detail: str = ""

@dataclass
class BasicBlock:
    block_id: int
    start: int
    end: int  # 半开区间
    instructions: List[DecodedInstruction] = field(default_factory=list)

    @property
    def address_range(self) -> Tuple[int, int]:
        return self.start, self.end

    def contains(self, addr: int) -> bool:
        return self.start <= addr < self.end


class CFAnalyzer:
    """在 :class:`DisassemblyDB` 之上执行控制流恢复。"""

    def __init__(self, db: DisassemblyDB) -> None:
        self.db = db
        self.entries: Set[int] = set()
        self._block_ids = count(0)
        self._edge_ids = count(0)
        # 已沿控制流到达并处理过的指令起点
        self.reached: Set[int] = set()
        self.edges: List[Edge] = []
        self._edge_keys: Set[Tuple[int, int, EdgeKind]] = set()
        self.unresolved: List[UnresolvedEdge] = []
        self._unresolved_keys: Set[Tuple[int, UnresolvedReason, str]] = set()
        self.blocks: List[BasicBlock] = []
        self.block_by_addr: Dict[int, BasicBlock] = {}
        # 被 CALL 指向的地址，即恢复出的函数入口
        self.call_targets: Set[int] = set()

    # ------------------------------------------------------------------ 遍历
    def analyze(self, *entries: int) -> None:
        queue: Deque[int] = deque()

        def enqueue(addr: int, is_entry: bool = False) -> None:
            if is_entry:
                self.entries.add(addr)
            if addr not in self.reached and addr not in queue:
                queue.append(addr)

        for entry in entries:
            if not self.db.space.is_mapped(entry):
                self._unresolved(
                    UnresolvedEdge(entry, UnresolvedReason.UNMAPPED_TARGET,
                                   "entry point is not mapped")
                )
                continue
            enqueue(entry, is_entry=True)

        while queue:
            address = queue.popleft()
            if address in self.reached:
                continue
            inst = self.db.decode_at(address)
            self.reached.add(address)
            if inst.status is not DecodeStatus.OK:
                self._unresolved(UnresolvedEdge(
                    address,
                    UnresolvedReason.DECODE_ERROR,
                    inst.status.value,
                ))
                continue
            self._expand(inst, enqueue)

        self._build_blocks()

    def _expand(
        self,
        inst: DecodedInstruction,
        enqueue,
    ) -> None:
        flow = inst.flow
        fall = inst.next_address

        if flow in (FlowKind.FALL, FlowKind.COND, FlowKind.CALL):
            self._connect_fall(inst.address, fall, enqueue)

        if flow in (FlowKind.COND, FlowKind.JUMP, FlowKind.CALL):
            target = inst.direct_target()
            assert target is not None
            kind = (
                EdgeKind.CALL_EDGE
                if flow is FlowKind.CALL
                else EdgeKind.BRANCH_TAKEN
            )
            if flow is FlowKind.CALL:
                self.call_targets.add(target)
            self._connect_direct(inst.address, target, kind, enqueue)

        if flow is FlowKind.INDIRECT:
            self._unresolved(UnresolvedEdge(
                inst.address,
                UnresolvedReason.INDIRECT_JUMP,
                f"target register operand={inst.operands}",
            ))
        elif flow is FlowKind.RETURN:
            self._unresolved(UnresolvedEdge(
                inst.address,
                UnresolvedReason.DYNAMIC_RETURN,
            ))
        # HALT：无后继，不登记任何边

    def _connect_fall(self, source: int, target: int, enqueue) -> None:
        if not self.db.space.is_mapped(target):
            self._unresolved(UnresolvedEdge(
                source,
                UnresolvedReason.UNMAPPED_TARGET,
                f"fallthrough into {target:#x}",
            ))
            return
        probe = self.db.peek(target)
        if probe.status is not DecodeStatus.OK:
            self._unresolved(UnresolvedEdge(
                source,
                UnresolvedReason.BAD_TARGET,
                f"fallthrough lands on {probe.status.value} at {target:#x}",
            ))
            return
        self._add_edge(source, target, EdgeKind.FALLTHROUGH)
        enqueue(target)

    def _connect_direct(self, source: int, target: int,
                        kind: EdgeKind, enqueue) -> None:
        if not self.db.space.is_mapped(target):
            reason = UnresolvedReason.UNMAPPED_TARGET
            self._unresolved(UnresolvedEdge(
                source, reason, f"direct target {target:#x} is not mapped"))
            return
        probe = self.db.peek(target)
        if probe.status is not DecodeStatus.OK:
            self._unresolved(UnresolvedEdge(
                source,
                UnresolvedReason.BAD_TARGET,
                f"target {target:#x} decodes as {probe.status.value}",
            ))
            return
        self._add_edge(source, target, kind)
        enqueue(target)

    def _add_edge(self, source: int, target: int, kind: EdgeKind) -> None:
        key = (source, target, kind)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(Edge(next(self._edge_ids), source, target, kind))

    def _unresolved(self, edge: UnresolvedEdge) -> None:
        key = (edge.source, edge.kind, edge.detail)
        if key not in self._unresolved_keys:
            self._unresolved_keys.add(key)
            self.unresolved.append(edge)

    # ------------------------------------------------------------------ 分块
    def _build_blocks(self) -> None:
        # 只有“沿控制流到达”的起点才属于指令序列；decode_at 的目标探测
        # 可能在别的指令中间实例化解码（重叠），不能污染基本块划分。
        ok = [
            inst
            for addr in sorted(self.reached)
            if (inst := self.db.instructions.get(addr)) is not None
            and inst.status is DecodeStatus.OK
        ]
        if not ok:
            return

        leaders: Set[int] = set(self.entries)
        for edge in self.edges:
            # 落空边连接每一对相邻指令，不能据此把每条指令都标成 leader
            if edge.kind is not EdgeKind.FALLTHROUGH:
                leaders.add(edge.target)
        for inst in ok:
            # 任何带终止语义的指令，其落空地址都是潜在新块起点
            if inst.flow in (FlowKind.COND, FlowKind.CALL):
                leaders.add(inst.next_address)

        current: Optional[BasicBlock] = None
        prev_end: Optional[int] = None
        for inst in ok:
            terminates_block = (
                current is not None
                and current.instructions
                and current.instructions[-1].flow is not FlowKind.FALL
            )
            starts_new = (
                inst.address in leaders
                or prev_end is None
                or inst.address != prev_end
                or terminates_block
            )
            if starts_new:
                current = BasicBlock(
                    block_id=next(self._block_ids),
                    start=inst.address,
                    end=inst.next_address,
                )
                self.blocks.append(current)
                self.block_by_addr[inst.address] = current
            assert current is not None
            current.instructions.append(inst)
            current.end = inst.next_address
            prev_end = inst.next_address

    # ------------------------------------------------------------------ 输出
    def block_at(self, addr: int) -> Optional[BasicBlock]:
        for block in self.blocks:
            if block.contains(addr):
                return block
        return None

    def successors(self, block: BasicBlock) -> List[Tuple[EdgeKind, BasicBlock]]:
        result: List[Tuple[EdgeKind, BasicBlock]] = []
        last = block.instructions[-1].address
        for edge in self.edges:
            if edge.source == last:
                target_block = self.block_by_addr.get(edge.target)
                if target_block is not None:
                    result.append((edge.kind, target_block))
        return result

    def summary(self) -> str:
        lines = [
            f"entries: {', '.join(hex(a) for a in sorted(self.entries)) or '-'}",
            f"reached instruction starts: {len(self.reached)}",
            f"basic blocks: {len(self.blocks)}",
            f"edges: {len(self.edges)}",
            f"unresolved: {len(self.unresolved)}",
        ]
        for edge in self.unresolved:
            lines.append(
                f"  UNRESOLVED {edge.source:#06x} {edge.kind.value}: {edge.detail}"
            )
        return "\n".join(lines)
