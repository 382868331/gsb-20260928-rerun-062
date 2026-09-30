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

INDIRECT_TABLE_SOURCE = """
entry:
  ldi r0, 0
  jmpr r1
  halt
table:
  .word target_a, target_b
  .word 0x1234
target_a:
  halt
target_b:
  ret
"""

RECURSIVE_CALL_SOURCE = """
entry:
  call countdown
  halt
countdown:
  jz r0, base
  subi r0, 1
  call countdown
  ret
base:
  ret

noreturn:
  jmp noreturn
"""


def branch_loop() -> AssembledProgram:
    return assemble(BRANCH_LOOP_SOURCE, origin=0)


def overlap() -> AssembledProgram:
    return assemble(OVERLAP_SOURCE, origin=0)


def indirect_table() -> AssembledProgram:
    return assemble(INDIRECT_TABLE_SOURCE, origin=0)


def recursive_call() -> AssembledProgram:
    return assemble(RECURSIVE_CALL_SOURCE, origin=0)
