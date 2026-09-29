"""Print analysis for the bundled synthetic teaching samples."""

from __future__ import annotations

import argparse

from .address import AddressSpace, Fragment, FragmentMapping
from .analysis import EdgeKind, TargetStatus, ControlFlowAnalyzer
from .samples import branch_loop, overlap


def build_result(sample: str):
    program = branch_loop() if sample == "branch-loop" else overlap()
    fragment = Fragment("builtin", sample, program.data)
    space = AddressSpace([FragmentMapping("builtin-map", fragment, 0)])
    return ControlFlowAnalyzer(space).analyze(program.symbols["entry"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a bundled teaching binary")
    parser.add_argument("--sample", choices=("branch-loop", "overlap"), default="branch-loop")
    result = build_result(parser.parse_args().sample)
    print("Instructions:")
    for instruction in result.instructions:
        print("  " + instruction.render())
    print("\nBlocks:")
    for block in result.blocks:
        suffix = " [overlap]" if block.overlap else ""
        print(f"  {block.block_id}: 0x{block.start_address:04x}-0x{block.end_address:04x}{suffix}")
    print("\nEdges:")
    for block in result.blocks:
        for edge in result.block_outgoing(block):
            target = "unresolved" if edge.target_address is None else f"0x{edge.target_address:04x}"
            print(f"  {block.block_id} {edge.kind.value}: {target} -> {edge.target_block_id or '-'}")
    print("\nTarget states:")
    for address, status in sorted(result.target_statuses.items()):
        print(f"  0x{address:04x}: {status.value}")
    if result.unresolved_indirect_jumps:
        print("\nIndirect jumps: " + ", ".join(result.unresolved_indirect_jumps))


if __name__ == "__main__":
    main()
