"""Small assembler used only to create teaching-format samples."""

from __future__ import annotations

from dataclasses import dataclass

from .isa import SPECS, Operation


class AssemblyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AssembledProgram:
    data: bytes
    symbols: dict[str, int]


def _register(token: str) -> int:
    token = token.strip()
    if not token.startswith("r"):
        raise AssemblyError(f"expected register, got {token!r}")
    try:
        value = int(token[1:], 10)
    except ValueError as exc:
        raise AssemblyError(f"invalid register {token!r}") from exc
    if not 0 <= value <= 255:
        raise AssemblyError(f"register out of range: {token}")
    return value


def _immediate(token: str) -> int:
    try:
        value = int(token.strip(), 0)
    except ValueError as exc:
        raise AssemblyError(f"invalid immediate {token!r}") from exc
    if not 0 <= value <= 255:
        raise AssemblyError(f"immediate out of range: {token}")
    return value


def assemble(source: str, origin: int = 0) -> AssembledProgram:
    statements: list[tuple[str, list[str], int]] = []
    symbols: dict[str, int] = {}
    addresses: list[int] = []
    address = origin
    for line_number, raw_line in enumerate(source.splitlines(), 1):
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            label = line[:-1].strip()
            if not label or label in symbols:
                raise AssemblyError(f"invalid or duplicate label on line {line_number}")
            symbols[label] = address
            continue
        parts = line.replace(",", " ").split()
        directive = parts[0].lower()
        if directive in {".byte", ".word"}:
            size = 1 if directive == ".byte" else 2
            statements.append((directive, parts[1:], line_number))
            addresses.append(address)
            address += max(1, len(parts) - 1) * size
            continue
        mnemonic = directive
        operations = {item.value: item for item in Operation}
        if mnemonic not in operations:
            raise AssemblyError(f"unknown mnemonic on line {line_number}: {mnemonic}")
        operation = operations[mnemonic]
        opcode = next(code for code, spec in SPECS.items() if spec.operation is operation)
        length = SPECS[opcode].length
        statements.append((mnemonic, parts[1:], line_number))
        addresses.append(address)
        address += length

    output = bytearray()
    for index, (mnemonic, operands, line_number) in enumerate(statements):
        if mnemonic in {".byte", ".word"}:
            size = 1 if mnemonic == ".byte" else 2
            if not operands:
                raise AssemblyError(f"{mnemonic} requires at least one value on line {line_number}")
            for token in operands:
                value = _data_value(token, symbols, line_number)
                if size == 1:
                    if not 0 <= value <= 0xFF:
                        raise AssemblyError(f".byte value out of range on line {line_number}")
                    output.append(value)
                else:
                    if not 0 <= value <= 0xFFFF:
                        raise AssemblyError(f".word value out of range on line {line_number}")
                    output.extend(((value >> 8) & 0xFF, value & 0xFF))
            assert len(output) - addresses[index] == len(operands) * size
            continue
        opcode = next(code for code, spec in SPECS.items() if spec.operation.value == mnemonic)
        output.append(opcode)
        if mnemonic in {"nop", "halt", "ret"}:
            if operands:
                raise AssemblyError(f"{mnemonic} takes no operands on line {line_number}")
        elif mnemonic in {"ldi", "subi"}:
            if len(operands) != 2:
                raise AssemblyError(f"{mnemonic} requires register and immediate on line {line_number}")
            output.extend((_register(operands[0]), _immediate(operands[1])))
        elif mnemonic in {"add", "sub"}:
            if len(operands) != 2:
                raise AssemblyError(f"{mnemonic} requires two registers on line {line_number}")
            output.extend((_register(operands[0]), _register(operands[1])))
        elif mnemonic == "jmpr":
            if len(operands) != 1:
                raise AssemblyError(f"jmpr requires one register on line {line_number}")
            output.append(_register(operands[0]))
        elif mnemonic in {"jmp", "call"}:
            if len(operands) != 1:
                raise AssemblyError(f"{mnemonic} requires one label on line {line_number}")
            target = _target(operands[0], symbols, origin + len(output), line_number)
            output.extend(((target >> 8) & 0xFF, target & 0xFF))
        elif mnemonic in {"jz", "jnz", "jl"}:
            if len(operands) != 2:
                raise AssemblyError(f"{mnemonic} requires register and label on line {line_number}")
            output.append(_register(operands[0]))
            target = _target(operands[1], symbols, origin + len(output), line_number)
            output.extend(((target >> 8) & 0xFF, target & 0xFF))
        assert len(output) - addresses[index] == SPECS[opcode].length
    return AssembledProgram(bytes(output), symbols)


def _target(token: str, symbols: dict[str, int], current_address: int, line_number: int) -> int:
    token = token.strip()
    if token in symbols:
        target = symbols[token]
    else:
        try:
            target = int(token, 0)
        except ValueError as exc:
            raise AssemblyError(f"undefined label {token!r} on line {line_number}") from exc
    if not 0 <= target <= 0xFFFF:
        raise AssemblyError(f"target out of range on line {line_number}")
    return target


def _data_value(token: str, symbols: dict[str, int], line_number: int) -> int:
    if token in symbols:
        return symbols[token]
    try:
        return int(token, 0)
    except ValueError as exc:
        raise AssemblyError(f"undefined data symbol {token!r} on line {line_number}") from exc
