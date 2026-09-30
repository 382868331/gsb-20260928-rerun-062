"""Print analysis for the bundled synthetic teaching samples."""

from __future__ import annotations

import argparse

from .address import AddressSpace, Fragment, FragmentMapping
from .analysis import AnalysisState, EdgeKind, TargetStatus, ControlFlowAnalyzer, JumpTable
from .samples import branch_loop, indirect_table, overlap


def build_result(sample: str):
    program = {
        "branch-loop": branch_loop(),
        "overlap": overlap(),
        "indirect-table": indirect_table(),
    }[sample]
    fragment = Fragment("builtin", sample, program.data)
    space = AddressSpace([FragmentMapping("builtin-map", fragment, 0)])
    tables = ()
    if sample == "indirect-table":
        tables = (JumpTable("builtin-table", program.symbols["entry"] + 3, program.symbols["table"], 3),)
    return ControlFlowAnalyzer(space).analyze(program.symbols["entry"], jump_tables=tables)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a bundled teaching binary")
    parser.add_argument("--sample", choices=("branch-loop", "overlap", "indirect-table"), default="branch-loop")
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
            print(f"  {block.block_id} {edge.kind.value} [{edge.state.value}]: {target} -> {edge.target_block_id or '-'}")
    print("\nTarget states:")
    for address, status in sorted(result.target_statuses.items()):
        print(f"  0x{address:04x}: {status.value}")
    if result.unresolved_indirect_jumps:
        print("\nIndirect jumps: " + ", ".join(result.unresolved_indirect_jumps))
    if result.indirect_candidates:
        print("\nIndirect candidates (not CFG edges):")
        for candidate in result.indirect_candidates:
            print(f"  {candidate.kind.value} 0x{candidate.target_address:04x} [{candidate.state.value}]")
    if result.procedures:
        print("\nProcedures:")
        for procedure in result.procedures:
            print(f"  0x{procedure.entry_address:04x}: may_return={procedure.may_return}")
    if result.analysis_state is not AnalysisState.COMPLETE:
        print(f"\nAnalysis state: {result.analysis_state.value} ({result.instructions_decoded} instructions decoded)")


if __name__ == "__main__":
    main()
