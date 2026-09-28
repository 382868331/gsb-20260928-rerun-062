"""文件片段、地址空间与稳定身份。

这一层刻意不包含任何“代码/数据”的判断：它只回答“某个地址上的字节
来自哪个片段的哪个文件偏移”。上层（反汇编、数据引用）通过这里分配的
稳定 ``id`` 引用同一个片段或段，后续增加数据引用分析时身份可直接沿用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class FileFragment:
    """一个二进制文件中的字节片段。

    片段不假设自己被加载到哪里：同一片段理论上可以映射到多个地址。
    ``fragment_id`` 在整个工作库内全局唯一且稳定。
    """

    fragment_id: int
    name: str
    data: bytes
    origin: str = "synthetic"  # synthetic（内置样本）或外部文件路径

    def __len__(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class Segment:
    """片段在地址空间中的一次映射：半开区间 [vaddr, vaddr+size)。"""

    segment_id: int
    fragment: FileFragment
    file_offset: int
    vaddr: int
    size: int
    writable: bool = False
    executable: bool = False

    @property
    def end(self) -> int:
        return self.vaddr + self.size

    def contains(self, addr: int) -> bool:
        return self.vaddr <= addr < self.end


class AddressSpace:
    """由若干互不重叠的段构成的平面地址空间。

    所有读操作都通过段与片段溯源，禁止靠“从头逐字节猜”越界读取。
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._frag_ids = count(0)
        self._seg_ids = count(0)
        self._fragments: Dict[int, FileFragment] = {}
        self._segments: Dict[int, Segment] = {}
        # 按 vaddr 排序的段列表，便于范围查询
        self._by_addr: List[Segment] = []

    # ------------------------------------------------------------------ 身份
    def add_fragment(
        self,
        data: bytes,
        name: str,
        origin: str = "synthetic",
    ) -> FileFragment:
        frag = FileFragment(next(self._frag_ids), name, bytes(data), origin)
        self._fragments[frag.fragment_id] = frag
        return frag

    def map_segment(
        self,
        fragment: FileFragment,
        vaddr: int,
        file_offset: int = 0,
        size: Optional[int] = None,
        *, writable: bool = False,
        executable: bool = False,
    ) -> Segment:
        if vaddr < 0:
            raise ValueError("vaddr 不能为负")
        if file_offset < 0 or file_offset > len(fragment):
            raise ValueError("file_offset 越界")
        if size is None:
            size = len(fragment) - file_offset
        if size < 0 or file_offset + size > len(fragment):
            raise ValueError("段大小超出片段范围")
        new_seg = Segment(
            next(self._seg_ids),
            fragment,
            file_offset,
            vaddr,
            size,
            writable,
            executable,
        )
        for seg in self._by_addr:
            if vaddr < seg.end and seg.vaddr < new_seg.end:
                raise ValueError(
                    f"段重叠: [{vaddr:#x},{new_seg.end:#x}) 与 "
                    f"[{seg.vaddr:#x},{seg.end:#x})"
                )
        self._segments[new_seg.segment_id] = new_seg
        self._by_addr.append(new_seg)
        self._by_addr.sort(key=lambda s: s.vaddr)
        return new_seg

    @property
    def fragments(self) -> Dict[int, FileFragment]:
        return dict(self._fragments)

    @property
    def segments(self) -> Dict[int, Segment]:
        return dict(self._segments)

    # ------------------------------------------------------------------ 查询
    def segment_at(self, addr: int) -> Optional[Segment]:
        for seg in self._by_addr:
            if seg.contains(addr):
                return seg
            if seg.vaddr > addr:
                break
        return None

    def is_mapped(self, addr: int) -> bool:
        return self.segment_at(addr) is not None

    def is_executable(self, addr: int) -> bool:
        seg = self.segment_at(addr)
        return seg is not None and seg.executable

    def provenance(self, addr: int) -> Optional[Tuple[FileFragment, int]]:
        """返回地址对应的片段与片段内偏移，供数据引用分析沿用身份。"""
        seg = self.segment_at(addr)
        if seg is None:
            return None
        return seg.fragment, seg.file_offset + (addr - seg.vaddr)

    def read(self, addr: int, length: int) -> Optional[bytes]:
        """连续读取 length 字节；跨段或未映射一律返回 None（不猜测）。"""
        if length < 0:
            raise ValueError("length 不能为负")
        seg = self.segment_at(addr)
        if seg is None:
            return None
        if addr + length > seg.end:
            return None
        start = seg.file_offset + (addr - seg.vaddr)
        return seg.fragment.data[start:start + length]

    def iter_mapped_ranges(self) -> Iterator[Tuple[int, int]]:
        for seg in self._by_addr:
            yield seg.vaddr, seg.end

    def next_mapped_or_end(self, addr: int) -> Optional[int]:
        """addr 落在未映射空洞时，返回下一个映射起点。"""
        for seg in self._by_addr:
            if seg.vaddr >= addr:
                return seg.vaddr
        return None
