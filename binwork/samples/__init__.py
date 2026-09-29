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
