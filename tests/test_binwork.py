import json
import unittest

from binwork.address import AddressSpace, Fragment, FragmentMapping
from binwork.analysis import (
    AnalysisResult,
    AnalysisBudget,
    AnalysisRevision,
    BranchChoice,
    CallOutcome,
    ConstraintKind,
    CandidateStatus,
    EdgeKind,
    EntryValueCondition,
    MissingBehaviorKind,
    QueryConstraints,
    PathStatus,
    ReachabilityStatus,
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

    def _constrained_program_result(self):
        # 0x00: jnz r0, 0x09
        # 0x04: call 0x0c
        # 0x07: nop
        # 0x08: halt
        # 0x09: jmp 0x04
        # 0x0c: halt (callee that does not return)
        data = b"".join([
            bytes([0x21, 0x00, 0x00, 0x09]),
            bytes([0x12, 0x00, 0x0c]),
            bytes([0x00]),
            bytes([0x01]),
            bytes([0x10, 0x00, 0x04]),
            bytes(4),
            bytes([0x01]),
        ])
        analyzer = ControlFlowAnalyzer(make_space(data), noreturn_addresses=(0x0c,))
        return analyzer.analyze(0)

    def test_branch_condition_makes_target_reachable_or_unreachable(self):
        result = self._constrained_program_result()

        taken = result.paths_to(
            0x04,
            constraints=QueryConstraints(
                entry_values=(EntryValueCondition(0, 1),),
                branches=(BranchChoice(0x00, ConstraintKind.BRANCH_TAKEN),),
            ),
        )
        not_taken = result.paths_to(
            0x04,
            constraints=QueryConstraints(
                entry_values=(EntryValueCondition(0, 0),),
                branches=(BranchChoice(0x00, ConstraintKind.BRANCH_NOT_TAKEN),),
            ),
        )

        self.assertEqual(taken.reachability, ReachabilityStatus.REACHABLE)
        self.assertEqual(not_taken.reachability, ReachabilityStatus.REACHABLE)
        self.assertEqual(
            [step.kind for step in taken.paths[0].steps],
            [EdgeKind.CONDITION_TAKEN, EdgeKind.DIRECT_BRANCH],
        )
        self.assertEqual(
            [step.kind for step in not_taken.paths[0].steps],
            [EdgeKind.CONDITION_NOT_TAKEN],
        )
        self.assertTrue(taken.paths[0].steps[0].evidence)
        self.assertNotEqual(taken.query_id, not_taken.query_id)

    def test_conflicting_conditions_report_basis_and_no_path(self):
        result = self._constrained_program_result()
        conditions = QueryConstraints(
            entry_values=(EntryValueCondition(0, 1), EntryValueCondition(0, 2)),
            branches=(
                BranchChoice(0x00, ConstraintKind.BRANCH_TAKEN),
                BranchChoice(0x00, ConstraintKind.BRANCH_NOT_TAKEN),
            ),
        )
        query = result.paths_to(0x04, constraints=conditions)

        self.assertEqual(query.reachability, ReachabilityStatus.CONFLICTING)
        self.assertEqual(query.paths, ())
        kinds = {item.kind for item in query.conflicts}
        self.assertIn("duplicate_entry_value", kinds)
        self.assertIn("contradictory_branch_choice", kinds)
        self.assertTrue(all(item.basis for item in query.conflicts))

    def test_call_outcome_conflicts_with_confirmed_noreturn(self):
        result = self._constrained_program_result()
        query = result.paths_to(
            0x08,
            constraints=QueryConstraints(
                calls=(CallOutcome(0x04, ConstraintKind.CALL_RETURNS),),
            ),
        )

        self.assertEqual(query.reachability, ReachabilityStatus.CONFLICTING)
        conflict = query.conflicts[0]
        self.assertEqual(conflict.kind, "call_outcome_contradicts_analysis")
        self.assertIn("callee_0x000c", conflict.basis[0])

    def test_noreturn_call_condition_prunes_return_successor(self):
        # call 0x07; halt; callee at 0x07 is a nop that reaches an unmapped byte
        data = bytes([0x12, 0x00, 0x07, 0x01, 0x00, 0x00, 0x00, 0xff])
        result = ControlFlowAnalyzer(make_space(data)).analyze(0)
        relationship = result.call_relationships[0]
        self.assertIsNone(relationship.callee_may_return)

        noreturn = result.paths_to(
            0x03,
            constraints=QueryConstraints(
                calls=(CallOutcome(0x00, ConstraintKind.CALL_NORETURN),),
            ),
        )
        returns = result.paths_to(
            0x03,
            constraints=QueryConstraints(
                calls=(CallOutcome(0x00, ConstraintKind.CALL_RETURNS),),
            ),
        )

        self.assertEqual(noreturn.reachability, ReachabilityStatus.UNREACHABLE)
        self.assertTrue(
            any(item.target_address == 0x03 for item in noreturn.excluded_segments)
        )
        self.assertEqual(returns.reachability, ReachabilityStatus.INSUFFICIENT)
        self.assertEqual(returns.paths[0].status, PathStatus.ASSUMED)
        self.assertEqual(
            returns.missing_behaviors[0].kind,
            MissingBehaviorKind.UNRESOLVED_CALL_RETURN,
        )
        self.assertIn("0x0000", returns.missing_behaviors[0].description)

    def test_compare_queries_reports_constraint_that_excludes_segment(self):
        result = self._constrained_program_result()
        taken = QueryConstraints(
            branches=(BranchChoice(0x00, ConstraintKind.BRANCH_TAKEN),)
        )
        not_taken = QueryConstraints(
            branches=(BranchChoice(0x00, ConstraintKind.BRANCH_NOT_TAKEN),)
        )

        comparison = result.compare_queries(0x07, taken, not_taken)
        only_taken = {(item.source_address, item.target_address) for item in comparison.excluded_only_in_a}
        only_not_taken = {(item.source_address, item.target_address) for item in comparison.excluded_only_in_b}

        self.assertIn((0x00, 0x04), only_taken)
        self.assertIn((0x00, 0x09), only_not_taken)
        self.assertTrue(all(item.reason for item in comparison.excluded_only_in_a))
        self.assertNotEqual(comparison.query_id_a, comparison.query_id_b)
        self.assertIn("excludes 0x0000->0x0004", comparison.explain())

    def test_budget_stop_leaves_unfinished_range_with_missing_behavior(self):
        # Recursion, a tail call into the jump area, and a partial table.
        data = b"".join([
            bytes([0x12, 0x00, 0x00]),  # call 0x00 (recursion)
            bytes([0x10, 0x00, 0x09]),  # jmp 0x09 (tail call into jump area)
            bytes(3),                   # padding 0x06..0x08
            bytes([0x11, 0x00]),       # jmpr r0 at 0x09
            (0x0e).to_bytes(2),       # one truncated table entry at 0x0b -> 0x0e
            bytes([0x00]),            # nop at 0x0d
            bytes([0x01]),            # halt at 0x0e
        ])
        analyzer = ControlFlowAnalyzer(
            make_space(data),
            known_indirect_targets=[KnownIndirectTarget(0x09, 0, 0x0d)],
            jump_tables=[JumpTableConstraint(0x0b, registers=(0,), entry_count=3, max_entries=3)],
        )
        # Stop immediately after the JMPR: known target 0x0d is queued and the
        # truncated table candidate 0x0e remains unfinished.
        first = analyzer.analyze(0, budget=AnalysisBudget(max_instructions=3))
        self.assertTrue(first.budget_exhausted)

        query = first.paths_to(0x0e, include_statuses=(PathStatus.UNRESOLVED,))

        self.assertEqual(query.reachability, ReachabilityStatus.INSUFFICIENT)
        kinds = {item.kind for item in query.missing_behaviors}
        self.assertIn(MissingBehaviorKind.QUEUED_TARGET, kinds)
        self.assertIn(MissingBehaviorKind.TRUNCATED_JUMP_TABLE, kinds)
        self.assertTrue(any(item.address == 0x0d for item in query.missing_behaviors))

        resumed = analyzer.resume(first, budget=AnalysisBudget(max_instructions=10))
        self.assertTrue(
            {item.instruction_id for item in first.instructions}
            <= {item.instruction_id for item in resumed.instructions}
        )
        self.assertTrue(first.pending_targets)

    def test_revised_condition_keeps_old_query_identity_and_proofs_independent(self):
        result = self._constrained_program_result()
        taken = QueryConstraints(
            label="old",
            branches=(BranchChoice(0x00, ConstraintKind.BRANCH_TAKEN),),
        )
        revised = taken.with_changes(
            branches=(BranchChoice(0x00, ConstraintKind.BRANCH_NOT_TAKEN),),
            label="new",
        )

        old_query = result.paths_to(0x0c, constraints=taken)
        new_query = result.paths_to(0x04, constraints=revised)

        self.assertNotEqual(old_query.query_id, new_query.query_id)
        self.assertEqual(old_query.paths[0].path_id[:5], "path:")
        self.assertTrue(new_query.paths)
        old_edges = {edge.edge_id for edge in result.edges}
        self.assertTrue(old_edges)

    def test_query_result_round_trips_with_condition_and_basis(self):
        result = self._constrained_program_result()
        query = result.paths_to(
            0x04,
            constraints=QueryConstraints(
                entry_values=(EntryValueCondition(0, 1),),
                branches=(BranchChoice(0x00, ConstraintKind.BRANCH_NOT_TAKEN),),
            ),
        )
        data = json.loads(query.to_json())

        self.assertEqual(data["reachability"], "reachable")
        self.assertTrue(data["query_id"].startswith("query:"))
        self.assertTrue(data["paths"][0]["steps"])
        self.assertTrue(data["paths"][0]["steps"][0]["evidence"])


if __name__ == "__main__":
    unittest.main()
