"""Finite eight-bit teaching instruction set.

Encoding:
  00 NOP
  01 HALT
  02 LDI Rd, imm8
  03 ADD Rd, Rs
  04 SUB Rd, Rs
  05 SUBI Rd, imm8
  10 JMP target16
  11 JMPR Rs              indirect jump
  12 CALL target16
  13 RET
  20 JZ Rd, target16
  21 JNZ Rd, target16
  22 JL Rd, target16

Multi-byte fields are unsigned big-endian. Jump targets may refer to
unmapped addresses or addresses that have not been analyzed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Operation(str, Enum):
    NOP = "nop"
    HALT = "halt"
    LDI = "ldi"
    ADD = "add"
    SUB = "sub"
    SUBI = "subi"
    JMP = "jmp"
    JMPR = "jmpr"
    CALL = "call"
    RET = "ret"
    JZ = "jz"
    JNZ = "jnz"
    JL = "jl"


class OperandKind(str, Enum):
    REGISTER = "register"
    IMMEDIATE = "immediate"
    ADDRESS = "address"


class BranchKind(str, Enum):
    NONE = "none"
    FALLTHROUGH = "fallthrough"
    UNCONDITIONAL = "unconditional"
    CONDITIONAL = "conditional"
    INDIRECT = "indirect"
    CALL = "call"
    RETURN = "return"


@dataclass(frozen=True, slots=True)
class Operand:
    kind: OperandKind
    value: int

    def render(self) -> str:
        if self.kind is OperandKind.REGISTER:
            return f"r{self.value}"
        if self.kind is OperandKind.ADDRESS:
            return f"0x{self.value:04x}"
        return str(self.value)


@dataclass(frozen=True, slots=True)
class InstructionSpec:
    opcode: int
    operation: Operation
    length: int
    branch_kind: BranchKind
    mnemonic: str


@dataclass(frozen=True, slots=True)
class Instruction:
    instruction_id: str
    address: int
    operation: Operation
    operands: tuple[Operand, ...]
    encoding: bytes
    next_address: int
    branch_kind: BranchKind
    branch_target: int | None = None
    truncated: bool = False
    unknown_opcode: int | None = None
    source_fragment_id: str | None = None

    @property
    def length(self) -> int:
        return len(self.encoding)

    @property
    def is_valid(self) -> bool:
        return self.unknown_opcode is None and not self.truncated

    @property
    def terminates_block(self) -> bool:
        return self.branch_kind in {
            BranchKind.UNCONDITIONAL,
            BranchKind.INDIRECT,
            BranchKind.RETURN,
        } or self.operation is Operation.HALT

    @property
    def mnemonic(self) -> str:
        if not self.is_valid:
            opcode = self.unknown_opcode if self.unknown_opcode is not None else self.encoding[0]
            return f".byte 0x{opcode:02x}"
        return self.operation.value

    def render(self) -> str:
        if not self.is_valid:
            note = "truncated" if self.truncated else "unknown opcode"
            return f"0x{self.address:04x}: {self.encoding.hex(' '):<8} ; {note}"
        operands = ", ".join(operand.render() for operand in self.operands)
        body = self.mnemonic if not operands else f"{self.mnemonic} {operands}"
        return f"0x{self.address:04x}: {self.encoding.hex(' '):<8} {body}"


REG = OperandKind.REGISTER
IMM = OperandKind.IMMEDIATE
ADDR = OperandKind.ADDRESS

SPECS: dict[int, InstructionSpec] = {
    0x00: InstructionSpec(0x00, Operation.NOP, 1, BranchKind.FALLTHROUGH, "nop"),
    0x01: InstructionSpec(0x01, Operation.HALT, 1, BranchKind.NONE, "halt"),
    0x02: InstructionSpec(0x02, Operation.LDI, 3, BranchKind.FALLTHROUGH, "ldi"),
    0x03: InstructionSpec(0x03, Operation.ADD, 3, BranchKind.FALLTHROUGH, "add"),
    0x04: InstructionSpec(0x04, Operation.SUB, 3, BranchKind.FALLTHROUGH, "sub"),
    0x05: InstructionSpec(0x05, Operation.SUBI, 3, BranchKind.FALLTHROUGH, "subi"),
    0x10: InstructionSpec(0x10, Operation.JMP, 3, BranchKind.UNCONDITIONAL, "jmp"),
    0x11: InstructionSpec(0x11, Operation.JMPR, 2, BranchKind.INDIRECT, "jmpr"),
    0x12: InstructionSpec(0x12, Operation.CALL, 3, BranchKind.CALL, "call"),
    0x13: InstructionSpec(0x13, Operation.RET, 1, BranchKind.RETURN, "ret"),
    0x20: InstructionSpec(0x20, Operation.JZ, 4, BranchKind.CONDITIONAL, "jz"),
    0x21: InstructionSpec(0x21, Operation.JNZ, 4, BranchKind.CONDITIONAL, "jnz"),
    0x22: InstructionSpec(0x22, Operation.JL, 4, BranchKind.CONDITIONAL, "jl"),
}


def _u16(data: bytes, offset: int) -> int:
    return (data[offset] << 8) | data[offset + 1]


def decode_instruction(
    address: int,
    data: bytes,
    instruction_id: str,
    fragment_id: str | None = None,
) -> Instruction:
    """Decode one instruction from the bytes beginning at ``address``."""
    if not data:
        raise ValueError("at least one byte is required to decode")
    opcode = data[0]
    spec = SPECS.get(opcode)
    if spec is None:
        return Instruction(
            instruction_id=instruction_id,
            address=address,
            operation=Operation.NOP,
            operands=(),
            encoding=data[:1],
            next_address=address + 1,
            branch_kind=BranchKind.NONE,
            unknown_opcode=opcode,
            source_fragment_id=fragment_id,
        )
    if len(data) < spec.length:
        return Instruction(
            instruction_id=instruction_id,
            address=address,
            operation=spec.operation,
            operands=(),
            encoding=bytes(data),
            next_address=address + len(data),
            branch_kind=spec.branch_kind,
            truncated=True,
            source_fragment_id=fragment_id,
        )

    operands: list[Operand] = []
    target: int | None = None
    if spec.operation in {Operation.LDI, Operation.SUBI}:
        operands = [Operand(REG, data[1]), Operand(IMM, data[2])]
    elif spec.operation in {Operation.ADD, Operation.SUB}:
        operands = [Operand(REG, data[1]), Operand(REG, data[2])]
    elif spec.operation is Operation.JMPR:
        operands = [Operand(REG, data[1])]
    elif spec.operation in {Operation.JMP, Operation.CALL}:
        target = _u16(data, 1)
        operands = [Operand(ADDR, target)]
    elif spec.branch_kind is BranchKind.CONDITIONAL:
        target = _u16(data, 2)
        operands = [Operand(REG, data[1]), Operand(ADDR, target)]

    return Instruction(
        instruction_id=instruction_id,
        address=address,
        operation=spec.operation,
        operands=tuple(operands),
        encoding=data[: spec.length],
        next_address=address + spec.length,
        branch_kind=spec.branch_kind,
        branch_target=target,
        source_fragment_id=fragment_id,
    )
