"""Built-in synthetic teaching binaries."""

from __future__ import annotations

from ..assembler import AssembledProgram, assemble

BRANCH_LOOP_SOURCE = """
entry:
  ldi r0, 0
  ldi r1, 10
loop:
  subi r1, 1
  add r0, r1
  jnz r1, loop
  jz r0, done
  ldi r2, 1
done:
  halt
"""

OVERLAP_SOURCE = """
entry:
  jmp middle
  ldi r1, 255
middle:
  add r0, r1
  halt
"""


def branch_loop() -> AssembledProgram:
    return assemble(BRANCH_LOOP_SOURCE, origin=0)


def overlap() -> AssembledProgram:
    return assemble(OVERLAP_SOURCE, origin=0)

RECURSIVE_TAIL_TABLE_SOURCE = """
entry:
  call recur
  jmpr r1
recur:
  jz r0, recur_done
  call recur
recur_done:
  jmp helper
helper:
  jmpr r2
after:
  call helper
  halt
"""


def recursive_tail_table() -> AssembledProgram:
    """Recursion, a tail call, and a partial jump table in one program.

    ``entry`` calls the recursive ``recur`` (which tail-jumps to the
    procedure entry ``helper``) and then dispatches through ``jmpr r1``.
    ``helper`` itself ends in an unresolved indirect jump, so whether any
    of these calls return is unknown until a call-behavior spec says so.
    Only two table entries are mapped even though the table is declared
    with three, so the entries stay truncated candidates.
    """
    program = assemble(RECURSIVE_TAIL_TABLE_SOURCE, origin=0)
    symbols = dict(program.symbols)
    symbols["table"] = len(program.data)
    data = (
        program.data
        + symbols["after"].to_bytes(2, "big")
        + symbols["helper"].to_bytes(2, "big")
    )
    return AssembledProgram(data, symbols)
