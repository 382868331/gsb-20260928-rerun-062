"""用 :mod:`teachbin.asm` 构造合成样本。

运行 ``python -m samples.build`` 会把样本写入 ``samples/bin/`` 供人工核对，
但分析与测试不依赖外部文件，字节在内存中直接生成。所有样本都是
teach-isa v1 字节，不加载、不运行任何真实可执行程序。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

from teachbin.asm import AssembledProgram, assemble
from teachbin.storage import AddressSpace


@dataclass
class Sample:
    name: str
    program: AssembledProgram
    entry: int
    vaddr: int
    extra_data: Dict[int, bytes] = field(default_factory=dict)

    def install(self, space: AddressSpace) -> None:
        frag = space.add_fragment(self.program.code, f"{self.name}.tb")
        space.map_segment(frag, self.vaddr, executable=True)
        for addr, blob in sorted(self.extra_data.items()):
            data_frag = space.add_fragment(blob, f"{self.name}.data@{addr:#x}")
            space.map_segment(data_frag, addr, writable=True)


def build_loop_sum() -> Sample:
    """1..10 求和：A=计数器, B=累加和, C=常量 10, D=临时。

    控制流含一个回边（循环）与一个条件退出边：

    ::

        A=1; B=0; C=10
      loop:
        CMP C              ; 标志位 = 计数器 A - 10
        JGT exit           ; 计数器 > 10 时退出
        D=A; A=B; ADD D    ; A = 和 + 计数器
        B=A; A=D           ; 保存新和，恢复计数器
        LOADI D,1; ADD D   ; 计数器 += 1
        JMP loop
      exit:
        A=B; mem[0]=A; HALT
    """
    program = assemble([
        "LOADI A, 1",
        "LOADI B, 0",
        "LOADI C, 10",
        "loop:",
        "CMP C",
        "JGT exit",
        "MOV D, A",
        "MOV A, B",
        "ADD D",
        "MOV B, A",
        "MOV A, D",
        "LOADI D, 1",
        "ADD D",
        "JMP loop",
        "exit:",
        "MOV A, B",
        "STORE A, 0",
        "HALT",
    ])
    return Sample("loop_sum", program, entry=0x1000, vaddr=0x1000,
                  extra_data={0x0000: bytes([0])})


def build_branch_call() -> Sample:
    """if/else 分支 + 对 helper 的 CALL/RET。

    ::

        A=5; B=0
        CMP B; JZ else     ; A==0 走 else，否则走 then
      then: A=1; JMP merge
      else: A=2
      merge: CALL helper   ; A += 7
        HALT
      helper:
        LOADI C,7; ADD C; RET
    """
    program = assemble([
        "LOADI A, 5",
        "LOADI B, 0",
        "CMP B",
        "JZ else",
        "LOADI A, 1",
        "JMP merge",
        "else:",
        "LOADI A, 2",
        "merge:",
        "CALL helper",
        "HALT",
        "helper:",
        "LOADI C, 7",
        "ADD C",
        "RET",
    ])
    return Sample("branch_call", program, entry=0x0400, vaddr=0x0400)


def build_unresolved_sample() -> Sample:
    """保留未解析状态：间接跳转 + 跳到未映射区域。"""
    program = assemble([
        "LOADI B, 0x80",
        "JMPIND B",
        "JMP +20",
    ])
    # 仅映射前 7 个字节，0x2002 起的 JMP 落在段内但其目标 0x201d 未映射
    trimmed = AssembledProgram(program.code[:7], program.labels)
    return Sample("unresolved", trimmed, entry=0x2000, vaddr=0x2000)


def build_truncated_fragment() -> Sample:
    """段内只有 0x10（LOADI 操作码），操作数被截断。"""
    program = AssembledProgram(code=bytes([0x10]), labels={})
    return Sample("truncated", program, entry=0x4000, vaddr=0x4000)


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "bin")
    os.makedirs(out_dir, exist_ok=True)
    builders = (
        build_loop_sum,
        build_branch_call,
        build_unresolved_sample,
        build_truncated_fragment,
    )
    for builder in builders:
        sample = builder()
        path = os.path.join(out_dir, f"{sample.name}.tb")
        with open(path, "wb") as fh:
            fh.write(sample.program.code)
        print(f"{path}: {len(sample.program.code)} bytes @ {sample.vaddr:#x}")
        for label, off in sample.program.labels.items():
            print(f"  {label} = {sample.vaddr + off:#06x}")


if __name__ == "__main__":
    main()
