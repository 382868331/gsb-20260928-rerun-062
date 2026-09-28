"""有限教学指令集 (teach-isa v1)。

变长指令，小端序。第一字节为操作码：

========  =====  ==============================================
操作码    长度  语义
========  =====  ==============================================
HALT      1     停机
NOP       1     空操作
LOADI r,i 3     r = imm8
ADD r     2     A = A + r
SUB r     2     A = A - r
CMP r     2     按 A-r 设置标志位
MOV d,s   3     d = s
STORE r,a 3     mem[imm8] = r
LOAD r,a  3     r = mem[imm8]
JMP rel   2     无条件相对跳转 (PC 为指令首址)
JZ rel    2     ZF=1 时相对跳转
JNZ rel   2     ZF=0 时相对跳转
JGT rel   2     大于时相对跳转
JMPIND r 2     间接跳转，目标 = regs[r]，静态不可确定
CALL rel  2     调用（相对调用点之后的返回地址）
RET       1     返回，目标动态确定
========  =====  ==============================================

相对偏移为有符号 8 位，目标地址 = 指令首址 + 指令长度 + rel。
寄存器编码 0..3：A=0, B=1, C=2, D=3。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


REGISTER_NAMES = ("A", "B", "C", "D")


class DecodeStatus(str, Enum):
    """单条指令解码的明确状态。"""

    OK = "ok"                # 完整解码
    UNKNOWN_OPCODE = "unknown_opcode"  # 操作码不在指令集中
    TRUNCATED = "truncated"  # 操作码有效但字节不足


class FlowKind(str, Enum):
    """指令对控制流的影响。"""

    FALL = "fall"            # 顺序执行
    JUMP = "jump"            # 无条件直接跳转
    COND = "cond"            # 条件直接跳转（有分支与落空两边）
    CALL = "call"            # 直接调用（调用后落空）
    INDIRECT = "indirect"    # 间接跳转，目标静态未知
    RETURN = "return"        # 返回，目标动态未知
    HALT = "halt"            # 终止，无后继


@dataclass(frozen=True)
class OpcodeSpec:
    opcode: int
    mnemonic: str
    length: int
    flow: FlowKind
    operand: str = ""  # "" | "reg" | "imm" | "rel" | "reg,reg" | "reg,imm"

    @property
    def has_relative_target(self) -> bool:
        return self.operand == "rel"

    @property
    def has_data_ref(self) -> bool:
        return self.operand in ("reg,imm",)  # LOAD/STORE 引用数据地址


def _specs() -> Dict[int, OpcodeSpec]:
    rows = [
        (0x00, "HALT", 1, FlowKind.HALT, ""),
        (0x01, "NOP", 1, FlowKind.FALL, ""),
        (0x10, "LOADI", 3, FlowKind.FALL, "reg,imm"),
        (0x11, "ADD", 2, FlowKind.FALL, "reg"),
        (0x12, "SUB", 2, FlowKind.FALL, "reg"),
        (0x13, "CMP", 2, FlowKind.FALL, "reg"),
        (0x14, "MOV", 3, FlowKind.FALL, "reg,reg"),
        (0x20, "STORE", 3, FlowKind.FALL, "reg,imm"),
        (0x21, "LOAD", 3, FlowKind.FALL, "reg,imm"),
        (0x30, "JMP", 2, FlowKind.JUMP, "rel"),
        (0x31, "JZ", 2, FlowKind.COND, "rel"),
        (0x32, "JNZ", 2, FlowKind.COND, "rel"),
        (0x33, "JGT", 2, FlowKind.COND, "rel"),
        (0x34, "JMPIND", 2, FlowKind.INDIRECT, "reg"),
        (0x40, "CALL", 2, FlowKind.CALL, "rel"),
        (0x41, "RET", 1, FlowKind.RETURN, ""),
    ]
    return {op: OpcodeSpec(op, m, ln, fl, opnd) for op, m, ln, fl, opnd in rows}


@dataclass(frozen=True)
class DecodedInstruction:
    address: int
    spec: Optional[OpcodeSpec]
    status: DecodeStatus
    raw: bytes
    operands: Tuple[int, ...] = ()
    # 解码状态非 OK 时用于记录“原本应有多长”
    expected_length: Optional[int] = None

    @property
    def size(self) -> int:
        return len(self.raw)

    @property
    def next_address(self) -> int:
        return self.address + self.size

    @property
    def flow(self) -> FlowKind:
        if self.spec is None:
            # 未知/截断指令按终止处理，避免凭空猜测后继
            return FlowKind.HALT
        return self.spec.flow

    def direct_target(self) -> Optional[int]:
        """直接分支/调用目标；其余指令返回 None。"""
        if self.spec is None or not self.spec.has_relative_target:
            return None
        if self.status is not DecodeStatus.OK:
            return None
        rel = _signed8(self.operands[0])
        return self.address + self.size + rel

    def data_reference(self) -> Optional[int]:
        """LOAD/STORE 的数据地址（绝对 imm8），为后续数据引用分析预留。"""
        if self.spec is None or not self.spec.has_data_ref:
            return None
        if self.status is not DecodeStatus.OK:
            return None
        return self.operands[1]

    def format(self) -> str:
        if self.status is DecodeStatus.UNKNOWN_OPCODE:
            return f".byte 0x{self.raw[0]:02x}  ; unknown opcode"
        if self.status is DecodeStatus.TRUNCATED:
            assert self.spec is not None
            hexd = " ".join(f"{b:02x}" for b in self.raw)
            return (
                f"{self.spec.mnemonic}  ; truncated "
                f"(need {self.expected_length}, got {self.size}: {hexd})"
            )
        assert self.spec is not None
        return _format_operands(self.spec, self.operands)

    def text(self) -> str:
        hexd = " ".join(f"{b:02x}" for b in self.raw)
        return f"{self.address:04x}: {hexd:<10} {self.format()}"


def _signed8(value: int) -> int:
    return value - 256 if value >= 128 else value


def _format_operands(spec: OpcodeSpec, ops: Sequence[int]) -> str:
    m = spec.mnemonic
    if spec.operand == "":
        return m
    if spec.operand == "reg":
        return f"{m} {REGISTER_NAMES[ops[0]]}"
    if spec.operand == "imm":
        return f"{m} {ops[0]}"
    if spec.operand == "rel":
        rel = _signed8(ops[0])
        sign = "+" if rel >= 0 else "-"
        return f"{m} {sign}{abs(rel)}"
    if spec.operand == "reg,reg":
        return f"{m} {REGISTER_NAMES[ops[0]]}, {REGISTER_NAMES[ops[1]]}"
    if spec.operand == "reg,imm":
        return f"{m} {REGISTER_NAMES[ops[0]]}, {ops[1]}"
    raise AssertionError(spec.operand)


class InstructionSet:
    """指令集单例式封装：只负责单条指令的纯解码，不做遍历。"""

    def __init__(self) -> None:
        self._specs = _specs()

    @property
    def opcode_map(self) -> Dict[int, OpcodeSpec]:
        return dict(self._specs)

    def spec(self, opcode: int) -> Optional[OpcodeSpec]:
        return self._specs.get(opcode)

    def decode_at(self, address: int, available: bytes) -> DecodedInstruction:
        """解码从 address 开始的一条指令。

        ``available`` 必须是地址空间在该地址实际可读的全部连续字节
        （长度可能为 0，例如地址未映射）。
        """
        if len(available) == 0:
            # 没有任何字节可读：按长度为 0 的截断指令处理，调用方决定如何记录
            return DecodedInstruction(
                address=address,
                spec=None,
                status=DecodeStatus.TRUNCATED,
                raw=b"",
                expected_length=None,
            )
        first = available[0]
        spec = self._specs.get(first)
        if spec is None:
            return DecodedInstruction(
                address=address,
                spec=None,
                status=DecodeStatus.UNKNOWN_OPCODE,
                raw=available[:1],
            )
        if len(available) < spec.length:
            return DecodedInstruction(
                address=address,
                spec=spec,
                status=DecodeStatus.TRUNCATED,
                raw=bytes(available),
                expected_length=spec.length,
            )
        raw = bytes(available[:spec.length])
        operands = tuple(raw[1:])
        return DecodedInstruction(
            address=address,
            spec=spec,
            status=DecodeStatus.OK,
            raw=raw,
            operands=operands,
        )
