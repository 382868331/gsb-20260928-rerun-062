import unittest

from binwork.address import AddressSpace, Fragment, FragmentMapping
from binwork.analysis import (
    AnalysisResult,
    AnalysisBudget,
    AnalysisRevision,
    BranchChoiceConstraint,
    CallResultConstraint,
    CandidateStatus,
    EntryValueConstraint,
    EdgeKind,
    PathConstraints,
    PathStatus,
    RevisionKind,
    JumpTableConstraint,
    KnownIndirectTarget,
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

    def test_resume_keeps_confirmed_candidate_and_pending_identities(self):
        data = bytes([0x00, 0x00, 0x10, 0x00, 0x05, 0x01, 0x00])
        analyzer = ControlFlowAnalyzer(make_space(data))
        first = analyzer.analyze(0, budget=AnalysisBudget(max_instructions=3))

        resumed = analyzer.resume(first, budget=AnalysisBudget(max_instructions=10))
        reloaded = AnalysisResult.from_json(first.to_json())

        self.assertTrue(first.budget_exhausted)
        self.assertFalse(resumed.budget_exhausted)
        self.assertEqual([item.target_id for item in reloaded.pending_targets], ["pending:0005"])
        self.assertEqual(reloaded.pending_targets[0].evidence[0].kind.value, "direct_operand")
        self.assertEqual(
            [item.instruction_id for item in resumed.instructions[:3]],
            [item.instruction_id for item in first.instructions],
        )
        self.assertEqual(resumed.target_statuses[5], TargetStatus.RESOLVED)

    def test_path_query_separates_confirmed_assumed_and_unresolved(self):
        data = b"".join([
            bytes([0x12, 0x00, 0x08]),
            bytes([0x10, 0x00, 0x0e]),
            bytes([0x13, 0x00]),
            bytes([0x12, 0x00, 0x08]),
            bytes([0x13, 0x00, 0x00]),
            bytes([0x11, 0x00]),
            bytes([0x00, 0x00, 0x01, 0x01]),
            (0x12).to_bytes(2),
        ])
        analyzer = ControlFlowAnalyzer(
            make_space(data),
            known_indirect_targets=[KnownIndirectTarget(0x0e, 0, 0x13)],
            jump_tables=[JumpTableConstraint(0x14, registers=(0,), entry_count=2, max_entries=2)],
        )
        result = analyzer.analyze(0)

        confirmed = result.paths_to(0x13, include_statuses=(PathStatus.CONFIRMED,))
        assumed = result.paths_to(0x12, include_statuses=(PathStatus.ASSUMED,))
        unresolved = result.paths_to(0x100, include_statuses=(PathStatus.UNRESOLVED,))

        self.assertEqual(confirmed.status, PathStatus.CONFIRMED)
        self.assertEqual(assumed.status, PathStatus.ASSUMED)
        self.assertEqual(unresolved.status, PathStatus.UNRESOLVED)
        kinds = [step.kind for step in confirmed.paths[0].steps]
        self.assertIn(EdgeKind.CALL_RETURN, kinds)
        self.assertTrue(any(step.evidence for step in confirmed.paths[0].steps))
        self.assertTrue(assumed.paths[0].assumption_ids)
        self.assertFalse(unresolved.paths[0].target_pending)

    def test_path_query_honors_entry_value_and_branch_choice_constraints(self):
        data = bytes([0x20, 0x00, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x01])
        result = ControlFlowAnalyzer(make_space(data)).analyze(0)

        zero = PathConstraints(
            entry_values=(EntryValueConstraint("entry-zero", 0, 0, 0),),
            branch_choices=(BranchChoiceConstraint("branch-taken", 0, True, 8),),
        )
        nonzero = PathConstraints(
            entry_values=(EntryValueConstraint("entry-nonzero", 0, 0, 1),),
            branch_choices=(BranchChoiceConstraint("branch-skip", 0, False, 4),),
        )

        taken = result.paths_to(8, constraints=zero)
        not_taken = result.paths_to(4, constraints=nonzero)
        comparison = result.compare_conditions(4, zero, nonzero)

        self.assertEqual(taken.status, PathStatus.REACHABLE)
        self.assertNotEqual(taken.query_id, not_taken.query_id)
        self.assertEqual(taken.paths[0].constraint_ids, zero.ids())
        self.assertTrue(taken.paths[0].steps[0].evidence)
        self.assertEqual(not_taken.status, PathStatus.REACHABLE)
        self.assertTrue(any(item.source_address == 0 for item in comparison.differences))
        self.assertTrue(any(item.right_included and not item.left_included for item in comparison.differences))

    def test_conflicting_constraints_report_basis_and_do_not_mix_paths(self):
        data = bytes([0x20, 0x00, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x01])
        result = ControlFlowAnalyzer(make_space(data)).analyze(0)
        conflicting = PathConstraints(
            entry_values=(EntryValueConstraint("entry-zero", 0, 0, 0),),
            branch_choices=(BranchChoiceConstraint("branch-not-taken", 0, False, 4),),
        )

        query = result.paths_to(4, constraints=conflicting)

        self.assertEqual(query.status, PathStatus.CONFLICT)
        self.assertEqual(query.paths, ())
        self.assertIn("entry-zero", query.conflicts[0].constraint_ids)
        self.assertIn("branch-not-taken", query.conflicts[0].constraint_ids)
        self.assertTrue(query.conflicts[0].evidence)

    def test_call_result_constraint_identifies_missing_behavior_and_no_return_path(self):
        data = bytes([0x12, 0x00, 0x06, 0x00, 0x00, 0x00])
        result = ControlFlowAnalyzer(make_space(data)).analyze(0)
        unknown = result.paths_to(3)
        no_return = PathConstraints(
            call_results=(CallResultConstraint("callee-no-return", False, 0, 6),),
        )

        query = result.paths_to(3, constraints=no_return)

        self.assertEqual(unknown.status, PathStatus.INSUFFICIENT)
        self.assertEqual(unknown.missing_behaviors[0].kind, "procedure_return")
        self.assertEqual(unknown.missing_behaviors[0].relation_id, "edge:ana.insn.0001:call_return:0003")
        self.assertEqual(query.status, PathStatus.UNREACHABLE)
        self.assertTrue(any(item.target_address == 3 for item in query.excluded_relations))

    def test_budget_leaves_unfinished_scope_for_recursion_tail_call_and_partial_table(self):
        data = b"".join([
            bytes([0x12, 0x00, 0x00]),
            bytes([0x10, 0x00, 0x06]),
            bytes([0x12, 0x00, 0x18]),
            bytes([0x10, 0x00, 0x00]),
            bytes([0x11, 0x00]),
            bytes([0x00]),
            bytes([0x00, 0x15, 0x00, 0x18, 0x00, 0x18, 0x00, 0x18]),
            bytes([0x13]),
            bytes([0x00, 0x00, 0x13]),
        ])
        analyzer = ControlFlowAnalyzer(
            make_space(data),
            jump_tables=[JumpTableConstraint(0x15, registers=(0,), entry_count=4, max_entries=4)],
        )
        result = analyzer.analyze(0, budget=AnalysisBudget(max_instructions=5))

        query = result.paths_to(0x20)

        self.assertTrue(result.budget_exhausted)
        self.assertEqual(query.status, PathStatus.INSUFFICIENT)
        self.assertTrue(any(item.kind == "pending_target" for item in query.unfinished_scopes))

    def test_partial_jump_table_revision_preserves_old_relationship_version(self):
        data = (
            bytes([0x11, 0x00, 0x00, 0x00])
            + (0x08).to_bytes(2)
            + bytes([0x00])
            + bytes([0x01, 0x00])
        )
        analyzer = ControlFlowAnalyzer(
            make_space(data),
            jump_tables=[JumpTableConstraint(0x04, registers=(0,), entry_count=2, max_entries=2)],
        )
        first = analyzer.analyze(0)
        completed = analyzer.revise_jump_table(
            first,
            JumpTableConstraint(
                0x04,
                registers=(0,),
                entry_count=2,
                max_entries=2,
                entries=(0x08, 0x07),
            ),
        )

        self.assertEqual(first.paths_to(0x09).status, PathStatus.UNRESOLVED)
        self.assertEqual(completed.result.paths_to(0x07).status, PathStatus.CONFIRMED)
        self.assertEqual(completed.revision.kind, RevisionKind.JUMP_TABLE)
        self.assertTrue(completed.revision.changed_relationship_ids)
        self.assertEqual(completed.previous_result, first)
        self.assertEqual(completed.result.revisions[0].revision_id, completed.revision.revision_id)

    def test_return_revision_recomputes_followup_and_keeps_old_version(self):
        data = bytes([0x12, 0x01, 0x00, 0x01])
        analyzer = ControlFlowAnalyzer(make_space(data))
        first = analyzer.analyze(0)
        revised = analyzer.revise_procedure_return(first, 0x100, False)

        self.assertIsNone(first.call_relationships[0].callee_may_return)
        self.assertFalse(revised.result.call_relationships[0].callee_may_return)
        self.assertTrue(any(edge.kind is EdgeKind.CALL_RETURN for edge in first.edges))
        self.assertFalse(any(edge.kind is EdgeKind.CALL_RETURN for edge in revised.result.edges))
        self.assertEqual(revised.revision.kind, RevisionKind.PROCEDURE_RETURN)


if __name__ == "__main__":
    unittest.main()
