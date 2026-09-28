"""反汇编数据库：记录解码结论，而不假设线性扫描正确。

任意地址都可以被请求解码（例如来自跳转目标），因此同一段字节可能因
不同解码起点产生重叠。本模块把这种情况显式记录为重叠冲突，而不是静默
覆盖；截断指令与未知操作码也保留为带状态的实体。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .isa import DecodeStatus, DecodedInstruction, InstructionSet
from .storage import AddressSpace, Segment


class ByteKind(str, Enum):
    """字节级分类，供展示与后续数据引用分析使用。"""

    UNMAPPED = "unmapped"
    UNANALYZED = "unanalyzed"          # 已映射但尚未被任何解码覆盖
    CODE_OK = "code_ok"
    CODE_OVERLAP = "code_overlap"      # 多个起点对同一字节解码且不一致
    CODE_UNKNOWN = "code_unknown"      # 属于未知操作码实体
    CODE_TRUNCATED = "code_truncated"  # 属于截断指令实体


@dataclass(frozen=True)
class OverlapConflict:
    address: int          # 发生冲突的字节地址
    owners: Tuple[int, ...]  # 主张该字节的各条指令起点


class DisassemblyDB:
    def __init__(self, space: AddressSpace, isa: Optional[InstructionSet] = None) -> None:
        self.space = space
        self.isa = isa or InstructionSet()
        # 指令实体（含 OK / 未知 / 截断），按起始地址索引
        self.instructions: Dict[int, DecodedInstruction] = {}
        # 字节地址 -> 覆盖它的指令起点集合
        self._coverers: Dict[int, Set[int]] = {}
        self._conflict_bytes: Set[int] = set()
        self._decode_attempts: Set[int] = set()

    # ------------------------------------------------------------------ 解码
    def decode_at(self, address: int) -> DecodedInstruction:
        """解码（或取回缓存）address 处的一条指令。

        相同起点的重复解码是幂等的；不同起点覆盖相同字节会产生冲突记录。
        """
        existing = self.instructions.get(address)
        if existing is not None:
            return existing

        inst = self._read_and_decode(address)
        self.instructions[address] = inst
        self._decode_attempts.add(address)
        self._record_coverage(inst)
        return inst

    def peek(self, address: int) -> DecodedInstruction:
        """非侵入式探测：不在数据库中登记指令或覆盖关系。

        控制流恢复在决定是否把一个目标加入工作队列前，用它检查目标处的
        字节状态；只有真正沿控制流到达的起点才会经 :meth:`decode_at`
        缓存，避免“探测”凭空制造重叠解码实体。
        """
        return self._read_and_decode(address)

    def _read_and_decode(self, address: int) -> DecodedInstruction:
        seg = self.space.segment_at(address)
        if seg is None:
            # 未映射地址也生成明确状态的空实体，由控制流层登记为未解析边
            return DecodedInstruction(
                address=address,
                spec=None,
                status=DecodeStatus.TRUNCATED,
                raw=b"",
                expected_length=None,
            )
        available = self.space.read(address, seg.end - address) or b""
        return self.isa.decode_at(address, available)

    def _record_coverage(self, inst: DecodedInstruction) -> None:
        if inst.size <= 0:
            return
        for addr in range(inst.address, inst.address + inst.size):
            owners = self._coverers.setdefault(addr, set())
            owners.add(inst.address)
            if len(owners) > 1:
                self._conflict_bytes.add(addr)

    # ------------------------------------------------------------------ 查询
    def was_decoded(self, address: int) -> bool:
        return address in self.instructions

    def ok_instructions(self) -> List[DecodedInstruction]:
        return [
            inst
            for inst in self.instructions.values()
            if inst.status is DecodeStatus.OK
        ]

    def by_status(self, status: DecodeStatus) -> List[DecodedInstruction]:
        return [
            inst
            for inst in self.instructions.values()
            if inst.status is status
        ]

    def truncated(self) -> List[DecodedInstruction]:
        return self.by_status(DecodeStatus.TRUNCATED)

    def unknown_opcodes(self) -> List[DecodedInstruction]:
        return self.by_status(DecodeStatus.UNKNOWN_OPCODE)

    def coverers(self, address: int) -> Tuple[int, ...]:
        return tuple(sorted(self._coverers.get(address, ())))

    def overlap_conflicts(self) -> List[OverlapConflict]:
        return [
            OverlapConflict(addr, tuple(sorted(self._coverers[addr])))
            for addr in sorted(self._conflict_bytes)
        ]

    def classify_byte(self, address: int) -> ByteKind:
        if not self.space.is_mapped(address):
            return ByteKind.UNMAPPED
        if address in self._conflict_bytes:
            return ByteKind.CODE_OVERLAP
        owner = next(iter(self._coverers.get(address, ())), None)
        if owner is None:
            return ByteKind.UNANALYZED
        inst = self.instructions[owner]
        if inst.status is DecodeStatus.OK:
            return ByteKind.CODE_OK
        if inst.status is DecodeStatus.UNKNOWN_OPCODE:
            return ByteKind.CODE_UNKNOWN
        return ByteKind.CODE_TRUNCATED

    def covered_ranges(self) -> List[Tuple[int, int]]:
        """所有已被至少一条解码实体覆盖的连续区间。"""
        if not self._coverers:
            return []
        addrs = sorted(self._coverers)
        ranges: List[Tuple[int, int]] = []
        start = prev = addrs[0]
        for addr in addrs[1:]:
            if addr == prev + 1:
                prev = addr
                continue
            ranges.append((start, prev + 1))
            start = prev = addr
        ranges.append((start, prev + 1))
        return ranges

    def unanalyzed_ranges(self) -> List[Tuple[int, int]]:
        """已映射但从未被解码覆盖的半开区间（含未分析的跳转目标区域）。"""
        covered = set(self._coverers)
        gaps: List[Tuple[int, int]] = []
        for vaddr, end in self.space.iter_mapped_ranges():
            gap_start: Optional[int] = None
            for addr in range(vaddr, end):
                if addr not in covered:
                    if gap_start is None:
                        gap_start = addr
                elif gap_start is not None:
                    gaps.append((gap_start, addr))
                    gap_start = None
            if gap_start is not None:
                gaps.append((gap_start, end))
        return gaps

    def listing(self) -> str:
        lines = []
        for addr in sorted(self.instructions):
            inst = self.instructions[addr]
            marker = ""
            if inst.status is DecodeStatus.TRUNCATED:
                marker = "  [TRUNCATED]"
            elif inst.status is DecodeStatus.UNKNOWN_OPCODE:
                marker = "  [UNKNOWN OPCODE]"
            if any(
                a in self._conflict_bytes
                for a in range(inst.address, inst.address + inst.size)
            ):
                marker += "  [OVERLAP]"
            lines.append(inst.text() + marker)
        return "\n".join(lines)
