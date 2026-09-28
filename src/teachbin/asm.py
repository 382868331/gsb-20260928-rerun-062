"""仅供构造教学样本的两遍汇编器。

它不读取、不解析、不运行任何真实可执行程序；输入是人类可读的助记符
列表（与测试/样本配套），输出是 teach-isa v1 字节。这样样本自带在
仓库里，分析结果可离线复核。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .isa import REGISTER_NAMES

_REGS = {name: i for i, name in enumerate(REGISTER_NAMES)}

# 助记符 -> (操作码, 长度)
_MNEMONICS: Dict[str, Tuple[int, int]] = {
    "HALT": (0x00, 1),
    "NOP": (0x01, 1),
    "LOADI": (0x10, 3),
    "ADD": (0x11, 2),
    "SUB": (0x12, 2),
    "CMP": (0x13, 2),
    "MOV": (0x14, 3),
    "STORE": (0x20, 3),
    "LOAD": (0x21, 3),
    "JMP": (0x30, 2),
    "JZ": (0x31, 2),
    "JNZ": (0x32, 2),
    "JGT": (0x33, 2),
    "JMPIND": (0x34, 2),
    "CALL": (0x40, 2),
    "RET": (0x41, 1),
}

_REL_MNEMONICS = {"JMP", "JZ", "JNZ", "JGT", "CALL"}


@dataclass
class AssembledProgram:
    code: bytes
    labels: Dict[str, int]  # 相对于代码起始的偏移


class AssemblyError(ValueError):
    pass


def _parse_register(token: str, line_no: int) -> int:
    token = token.strip()
    if token not in _REGS:
        raise AssemblyError(f"第 {line_no} 行: 未知寄存器 {token!r}")
    return _REGS[token]


def _parse_imm(token: str, line_no: int) -> int:
    token = token.strip()
    try:
        value = int(token, 0)
    except ValueError as exc:
        raise AssemblyError(f"第 {line_no} 行: 非法立即数 {token!r}") from exc
    if not 0 <= value <= 255:
        raise AssemblyError(f"第 {line_no} 行: 立即数超出 0..255: {value}")
    return value


def _split_operands(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def assemble(lines: Sequence[str]) -> AssembledProgram:
    """两遍汇编。

    标签独占一行或以冒号结尾，例如 ``loop:``；跳转操作数可以是标签名或
    绝对数字。数字立即数支持 0x 前缀。
    """
    # 第一遍：计算每条指令的地址与标签
    items: List[Tuple[int, str, List[str]]] = []  # (addr, mnemonic, operands)
    labels: Dict[str, int] = {}
    addr = 0
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            label = line[:-1].strip()
            if not label or label in labels:
                raise AssemblyError(f"第 {idx} 行: 标签非法或重复 {label!r}")
            labels[label] = addr
            continue
        head, _, rest = line.partition(" ")
        head = head.upper()
        if head not in _MNEMONICS:
            raise AssemblyError(f"第 {idx} 行: 未知助记符 {head!r}")
        operands = _split_operands(rest) if rest.strip() else []
        items.append((addr, head, operands))
        addr += _MNEMONICS[head][1]

    # 第二遍：编码
    out = bytearray()
    for inst_addr, mnemonic, operands in items:
        opcode, length = _MNEMONICS[mnemonic]
        out.append(opcode)
        if mnemonic in ("HALT", "NOP", "RET"):
            expected = 0
        elif mnemonic in ("ADD", "SUB", "CMP", "JMPIND"):
            expected = 1
            out.append(_parse_register(operands[0], 0))
        elif mnemonic in _REL_MNEMONICS:
            expected = 1
            token = operands[0].strip()
            if token.lstrip("-").isalpha() or token in labels:
                if token not in labels:
                    raise AssemblyError(f"未定义标签 {token!r}")
                rel = labels[token] - (inst_addr + length)
            else:
                rel = int(token, 0)
            if not -128 <= rel <= 127:
                raise AssemblyError(
                    f"{mnemonic}@{inst_addr}: 相对偏移 {rel} 超出有符号 8 位"
                )
            out.append(rel & 0xFF)
        elif mnemonic in ("LOADI", "STORE", "LOAD"):
            expected = 2
            out.append(_parse_register(operands[0], 0))
            out.append(_parse_imm(operands[1], 0))
        elif mnemonic == "MOV":
            expected = 2
            out.append(_parse_register(operands[0], 0))
            out.append(_parse_register(operands[1], 0))
        else:  # pragma: no cover - 表已穷举
            raise AssertionError(mnemonic)
        if len(operands) != expected:
            raise AssemblyError(
                f"{mnemonic}@{inst_addr}: 需要 {expected} 个操作数，得到 {len(operands)}"
            )
    return AssembledProgram(code=bytes(out), labels=labels)
