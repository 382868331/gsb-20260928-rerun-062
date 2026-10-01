import unittest

from binwork.address import AddressSpace, Fragment, FragmentMapping
from binwork.analysis import (
    AnalysisResult,
    AnalysisBudget,
    CandidateStatus,
    EdgeKind,
    JumpTableConstraint,
    KnownIndirectTarget,
    PathStatus,
    TargetStatus,
    ControlFlowAnalyzer,
)
from binwork.isa import Operation, decode_instruction
from binwork.samples import branch_loop, overlap


def make_space(data: bytes, load_address: int = 0, fragment_id: str = "frag") -> AddressSpace:
    fragment = Fragment(fragment_id, "synthetic", data)
    return AddressSpace([FragmentMapping("map", fragment, load_address)])


class FragmentAddressSpaceTests(unittest.TestCase):
    def test_fragment_identity_separate_from_mapping(self):
        fragment = Fragment("sample", "sample", b"abc")
        first = FragmentMapping("map-a", fragment, 0x10)
        second = FragmentMapping("map-b", fragment, 0x80)
        space = AddressSpace([first, second])

        self.assertEqual(space.read(0x10, 3).data, b"abc")
        self.assertEqual(space.read(0x80, 3).data, b"abc")
        self.assertEqual(space.location_at(0x81).fragment_id, "sample")
        self.assertEqual(space.location_at(0x81).fragment_offset, 1)

    def test_read_stops_at_unmapped_gap_and_reports_address(self):
        space = make_space(b"\x00\x00", 0x10)
        result = space.read(0x10, 4)
        self.assertFalse(result.complete)
        self.assertEqual(result.missing_from, 0x12)
        self.assertEqual(result.data, b"\x00\x00")


class IsolatedDecodeTests(unittest.TestCase):
    def test_decode_three_byte_add(self):
        instruction = decode_instruction(0x20, bytes([0x03, 2, 3]), "insn")
        self.assertEqual(instruction.operation, Operation.ADD)
        self.assertEqual(instruction.next_address, 0x23)
        self.assertEqual(instruction.render(), "0x0020: 03 02 03 add r2, r3")

    def test_truncated_instruction_is_retained(self):
        instruction = decode_instruction(0x30, bytes([0x10, 0x01]), "insn")
        self.assertTrue(instruction.truncated)
        self.assertEqual(instruction.encoding, b"\x10\x01")
        self.assertEqual(instruction.next_address, 0x32)

    def test_unknown_opcode_is_explicit(self):
        instruction = decode_instruction(0, bytes([0xff]), "insn")
        self.assertEqual(instruction.unknown_opcode, 0xff)
        self.assertFalse(instruction.is_valid)


class ControlFlowTests(unittest.TestCase):
    def test_builtin_branch_loop_path(self):
        program = branch_loop()
        result = ControlFlowAnalyzer(make_space(program.data)).analyze(program.symbols["entry"])

        addresses = [insn.address for insn in result.instructions]
        self.assertEqual(addresses[0], 0)
        self.assertIn(program.symbols["loop"], addresses)
        self.assertEqual(result.target_statuses[program.symbols["loop"]], TargetStatus.RESOLVED)
        self.assertEqual(result.target_statuses[program.symbols["done"]], TargetStatus.RESOLVED)

        taken = {edge.target_address: edge for edge in result.edges if edge.kind is EdgeKind.CONDITION_TAKEN}
        self.assertEqual(taken[program.symbols["loop"]].target_block_id, "ana.block.002")
        halt = next(insn for insn in result.instructions if insn.operation is Operation.HALT)
        self.assertEqual(halt.address, program.symbols["done"])
        self.assertTrue(all(block.start_address <= ins.address < block.end_address for block in result.blocks for ins in [result.instruction(block.instruction_ids[0])]))

    def test_future_target_is_discovered_without_linear_scan(self):
        program = overlap()
        symbols = overlap().symbols
        space = make_space(program.data)
        result = ControlFlowAnalyzer(space).analyze(symbols["entry"])

        decoded = {insn.address: insn.operation for insn in result.instructions}
        self.assertNotIn(3, decoded)
        self.assertEqual(decoded[6], Operation.ADD)
        self.assertEqual(decoded[9], Operation.HALT)

    def test_overlapping_decode_keeps_both_identities(self):
        data = bytes([0x10, 0x00, 0x02, 0x02, 0x01, 0x00])
        space = make_space(data)
        result = ControlFlowAnalyzer(space).analyze(0, 2)

        at_two = [insn for insn in result.instructions if insn.address == 2]
        self.assertEqual(len(at_two), 1)
        self.assertEqual(at_two[0].operation, Operation.LDI)
        at_zero = [insn for insn in result.instructions if insn.address == 0]
        self.assertEqual(len(at_zero), 1)
        self.assertTrue(any(block.overlap for block in result.blocks))

    def test_unmapped_direct_target_is_status_not_exception(self):
        space = make_space(bytes([0x10, 0x12, 0x34]))
        result = ControlFlowAnalyzer(space).analyze(0)
        self.assertEqual(result.target_statuses[0x1234], TargetStatus.UNMAPPED)
        edge = next(edge for edge in result.edges if edge.target_address == 0x1234)
        self.assertFalse(edge.resolved)

    def test_truncated_and_indirect_have_explicit_states(self):
        space = make_space(bytes([0x11, 0x01, 0x20, 0x00]))
        result = ControlFlowAnalyzer(space).analyze(0, 2)

        self.assertEqual(result.unresolved_indirect_jumps, ("ana.insn.0001",))
        self.assertIn("ana.insn.0002", {insn.instruction_id for insn in result.instructions})
        truncated = result.instruction("ana.insn.0002")
        self.assertTrue(truncated.truncated)
        indirect_edge = next(edge for edge in result.edges if edge.kind is EdgeKind.INDIRECT)
        self.assertIsNone(indirect_edge.target_address)
        self.assertFalse(indirect_edge.resolved)

    def test_unknown_opcode_does_not_create_fallthrough(self):
        space = make_space(bytes([0xff, 0x00]))
        result = ControlFlowAnalyzer(space).analyze(0)
        self.assertEqual(len(result.instructions), 1)
        self.assertEqual(result.instructions[0].unknown_opcode, 0xff)
        self.assertEqual(result.edges, ())

    def test_bounded_jump_table_confirms_only_exact_entries(self):
        data = bytes([0x11, 0x00, 0x00, 0x00]) + (0x00).to_bytes(2) + (0x08).to_bytes(2) + bytes([0x01, 0x00])
        space = make_space(data)
        analyzer = ControlFlowAnalyzer(
            space,
            jump_tables=[JumpTableConstraint(table_address=4, registers=(0,), entry_count=2, max_entries=2)],
        )

        result = analyzer.analyze(0)

        targets = {edge.target_address for edge in result.edges if edge.kind is EdgeKind.INDIRECT}
        self.assertEqual(targets, {0, 8})
        self.assertEqual(result.unresolved_indirect_jumps, ())
        self.assertEqual({ins.address for ins in result.instructions}, {0, 8})
        evidence = {edge.evidence[0].kind.value for edge in result.edges if edge.kind is EdgeKind.INDIRECT}
        self.assertEqual(evidence, {"jump_table"})

    def test_truncated_table_keeps_candidates_without_promotion(self):
        data = bytes([0x11, 0x00, 0x00, 0x00]) + (0x00).to_bytes(2) + bytes([0x00])
        space = make_space(data)
        analyzer = ControlFlowAnalyzer(
            space,
            jump_tables=[JumpTableConstraint(table_address=4, registers=(0,), entry_count=2, max_entries=2)],
        )

        result = analyzer.analyze(0)
        reloaded = AnalysisResult.from_json(result.to_json())

        self.assertEqual([candidate.target_address for candidate in result.indirect_target_candidates], [0])
        self.assertTrue(
            all(candidate.status is CandidateStatus.TRUNCATED for candidate in result.indirect_target_candidates)
        )
        self.assertFalse(any(edge.resolved for edge in result.edges if edge.kind is EdgeKind.INDIRECT))
        self.assertEqual(result.unresolved_indirect_jumps, ("ana.insn.0001",))
        self.assertEqual(reloaded.indirect_target_candidates, result.indirect_target_candidates)
        self.assertEqual(reloaded.confirmed_edges(), result.confirmed_edges())

    def test_known_constant_can_resolve_indirect_jump(self):
        space = make_space(bytes([0x11, 0x02, 0x00, 0x01]))
        analyzer = ControlFlowAnalyzer(
            space,
            known_indirect_targets=[KnownIndirectTarget(0, 2, 3)],
        )

        result = analyzer.analyze(0)

        edge = next(edge for edge in result.edges if edge.kind is EdgeKind.INDIRECT)
        self.assertTrue(edge.resolved)
        self.assertEqual(edge.target_address, 3)
        self.assertEqual(edge.evidence[0].kind.value, "known_constant")

    def test_recursive_call_is_returnable(self):
        data = bytes([
            0x12, 0x00, 0x00,
            0x20, 0x00, 0x00, 0x08,
            0x00,
            0x13,
            0x13,
        ])
        result = ControlFlowAnalyzer(make_space(data)).analyze(0)

        recursive_call = next(edge for edge in result.edges if edge.kind is EdgeKind.CALL and edge.target_address == 0)
        return_edge = next(edge for edge in result.edges if edge.kind is EdgeKind.CALL_RETURN)
        self.assertEqual(recursive_call.evidence[0].kind.value, "call_operand")
        self.assertTrue(return_edge.resolved)
        self.assertEqual(result.procedures[0].may_return, True)

    def test_noreturn_call_has_no_return_edge(self):
        data = bytes([0x12, 0x00, 0x04, 0x00, 0x00, 0x01, 0x00])
        analyzer = ControlFlowAnalyzer(make_space(data), noreturn_addresses=(4,))

        result = analyzer.analyze(0)

        self.assertFalse(any(edge.kind is EdgeKind.CALL_RETURN for edge in result.edges))
        relationship = result.call_relationships[0]
        self.assertEqual(relationship.callee_address, 4)
        self.assertFalse(relationship.callee_may_return)
        self.assertNotIn(3, {ins.address for ins in result.instructions})

    def test_tail_call_has_no_static_return_successor(self):
        data = bytes([0x12, 0x00, 0x04, 0x01, 0x10, 0x00, 0x07, 0x13])
        result = ControlFlowAnalyzer(make_space(data)).analyze(0)

        relationship = next(item for item in result.call_relationships if item.tail_call)
        edge = next(edge for edge in result.edges if edge.kind is EdgeKind.TAIL_CALL)
        self.assertEqual(relationship.callee_address, 7)
        self.assertIsNone(relationship.return_address)
        self.assertEqual(edge.target_address, 7)
        self.assertEqual(edge.evidence[0].kind.value, "tail_call")

    def test_budget_exhaustion_retains_confirmed_partial_result(self):
        data = bytes([0x00, 0x00, 0x10, 0x00, 0x05, 0x01, 0x00])
        result = ControlFlowAnalyzer(make_space(data)).analyze(0, budget=AnalysisBudget(max_instructions=3))

        self.assertTrue(result.budget_exhausted)
        self.assertEqual([ins.address for ins in result.instructions], [0, 1, 2])
        self.assertEqual(result.target_statuses[5], TargetStatus.QUEUED)
        self.assertTrue(any(edge.resolved for edge in result.edges))
        reloaded = AnalysisResult.from_json(result.to_json())
        self.assertTrue(reloaded.budget_exhausted)
        self.assertEqual([ins.address for ins in reloaded.instructions], [0, 1, 2])

    def test_budget_resume_keeps_identities_and_processes_pending_targets(self):
        data = bytes([0x00, 0x00, 0x10, 0x00, 0x05, 0x01, 0x00])
        analyzer = ControlFlowAnalyzer(make_space(data))
        partial = analyzer.analyze(0, budget=AnalysisBudget(max_instructions=3))

        resumed = analyzer.continue_analysis(partial)

        self.assertFalse(resumed.budget_exhausted)
        self.assertEqual(resumed.pending_targets(), ())
        self.assertTrue(
            {ins.instruction_id for ins in partial.instructions}
            <= {ins.instruction_id for ins in resumed.instructions}
        )
        self.assertIn(5, {ins.address for ins in resumed.instructions})

    def test_reachable_query_distinguishes_confirmed_assumed_and_unresolved(self):
        data = bytes([
            0x12, 0x00, 0x08,
            0x01,
            0x00, 0x00, 0x00, 0x00,
            0x11, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x10, 0x00, 0x0e,
            0x00, 0x00, 0x00,
            0x00, 0x0e,
            0x00, 0x17,
            0x00,
        ])
        analyzer = ControlFlowAnalyzer(
            make_space(data),
            jump_tables=[JumpTableConstraint(table_address=20, registers=(0,), entry_count=3, max_entries=3)],
        )
        result = analyzer.analyze(0)

        unknown_return = result.path_to_address(3)
        assumed_return = result.path_to_address(3, assume_callee_returns=True)
        assumed_candidate = result.path_to_address(23, assume_candidates=True)

        self.assertIs(unknown_return.status, PathStatus.UNRESOLVED)
        self.assertTrue(unknown_return.evidence)
        self.assertIs(assumed_return.status, PathStatus.ASSUMED)
        self.assertEqual(assumed_return.assumptions[0].kind, "callee_returns")
        self.assertIs(assumed_candidate.status, PathStatus.ASSUMED)
        self.assertEqual(assumed_candidate.target_address, 23)
        self.assertEqual(assumed_candidate.assumptions[0].kind, "indirect_target_candidate")

    def test_completing_jump_table_recomputes_with_old_relationships_retained(self):
        data = bytes([
            0x12, 0x00, 0x08,
            0x01,
            0x00, 0x00, 0x00, 0x00,
            0x11, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x10, 0x00, 0x0e,
            0x00, 0x00, 0x00,
            0x00, 0x0e,
            0x00,
        ])
        space = make_space(data)
        partial_table = JumpTableConstraint(table_address=20, registers=(0,), entry_count=2, max_entries=2)
        analyzer = ControlFlowAnalyzer(space, jump_tables=[partial_table])
        initial = analyzer.analyze(0)
        initial_call = next(item for item in initial.call_relationships if item.callee_address == 8)

        self.assertIsNone(initial_call.callee_may_return)
        self.assertTrue(any(edge.kind is EdgeKind.CALL_RETURN for edge in initial.edges))
        self.assertEqual(initial.indirect_target_candidates[0].status, CandidateStatus.TRUNCATED)

        revised_analyzer = ControlFlowAnalyzer(
            space,
            jump_tables=[JumpTableConstraint(table_address=20, registers=(0,), entry_count=1, max_entries=1)],
        )
        revised = revised_analyzer.revise(initial, detail="complete one-entry dispatch table")
        revised_call = next(item for item in revised.call_relationships if item.callee_address == 8)

        self.assertEqual(revised.revision, 1)
        self.assertFalse(revised_call.callee_may_return)
        self.assertFalse(any(edge.kind is EdgeKind.CALL_RETURN for edge in revised.edges))
        self.assertEqual(revised.indirect_target_candidates[0].status, CandidateStatus.CANDIDATE)
        self.assertEqual(revised.relationship_history[0].call_relationships, initial.call_relationships)
        self.assertEqual(revised.relationship_history[0].edges, initial.edges)
        reloaded = AnalysisResult.from_json(revised.to_json())
        self.assertEqual(reloaded.relationship_history[0].detail, "complete one-entry dispatch table")

    def test_return_revision_recomputes_affected_call_edges(self):
        data = bytes([0x12, 0x00, 0x04, 0x01, 0x10, 0x00, 0x04])
        analyzer = ControlFlowAnalyzer(make_space(data))
        initial = analyzer.analyze(0)
        initial_call = initial.call_relationships[0]

        self.assertIsNone(initial_call.callee_may_return)
        revised_analyzer = ControlFlowAnalyzer(make_space(data), may_return_overrides={4: True})
        revised = revised_analyzer.revise(initial, detail="callee returns", may_return_overrides={4: True})

        self.assertTrue(revised.call_relationships[0].callee_may_return)
        return_edge = next(edge for edge in revised.edges if edge.kind is EdgeKind.CALL_RETURN)
        self.assertTrue(return_edge.resolved)
        self.assertEqual(revised.relationship_history[0].revision, 0)


if __name__ == "__main__":
    unittest.main()
