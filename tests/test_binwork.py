import unittest

from binwork.address import AddressSpace, Fragment, FragmentMapping
from binwork.analysis import (
    AnalysisResult,
    AnalysisBudget,
    AnalysisRevision,
    CallBehaviorRegistry,
    CallBehaviorSpec,
    ContinuationKind,
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
from binwork.analysis import SpecApplicability
from binwork.analysis import PartitionCategory, PartitionReport
from binwork.samples import branch_loop, input_dependent_call, overlap, recursive_tail_table


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

class CallBehaviorVersionTests(unittest.TestCase):
    def unknown_callee_program(self):
        # call sub (0) ; halt (3) ; sub: jmpr r0 (4)
        return bytes([0x12, 0x00, 0x04, 0x01, 0x11, 0x00])

    def test_query_conclusion_names_adopted_spec_version(self):
        analyzer = ControlFlowAnalyzer(make_space(self.unknown_callee_program()))
        first = analyzer.analyze(0)
        old_query = first.paths_to(3)
        self.assertEqual(old_query.status, PathStatus.INSUFFICIENT)

        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec("sub-return", 1, callee_address=4, may_return=True)
        )
        report = analyzer.revise_call_behavior(first, registry, queries=(old_query,))

        query = report.result.paths_to(3)
        self.assertEqual(query.status, PathStatus.CONFIRMED)
        self.assertIn("sub-return@v1", query.behavior_refs)
        return_step = next(
            step for path in query.paths for step in path.steps if step.kind is EdgeKind.CALL_RETURN
        )
        self.assertEqual(return_step.evidence[0].kind.value, "call_behavior_spec")
        self.assertIn("sub-return@v1", return_step.behavior_refs)
        self.assertEqual(report.revision.kind, RevisionKind.CALL_BEHAVIOR)
        impact = report.query_impacts[0]
        self.assertTrue(impact.depends_on_changed_spec)
        self.assertEqual(impact.previous_status, PathStatus.INSUFFICIENT)
        self.assertEqual(impact.status, PathStatus.CONFIRMED)
        self.assertEqual(first.paths_to(3).status, PathStatus.INSUFFICIENT)

    def test_revision_preview_scopes_affected_and_unaffected_proofs(self):
        # call a (0); call b (3); halt (6); a: jmpr r0 (7); b: ret (9)
        data = bytes([0x12, 0x00, 0x07, 0x12, 0x00, 0x09, 0x01, 0x11, 0x00, 0x13])
        analyzer = ControlFlowAnalyzer(make_space(data))
        first = analyzer.analyze(0)
        query_b = first.paths_to(6)
        self.assertEqual(query_b.status, PathStatus.INSUFFICIENT)

        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec("a-return", 1, callee_address=7, may_return=True)
        )
        report = analyzer.revise_call_behavior(first, registry, queries=(query_b,))

        self.assertIn("a-return@v1", report.changed_spec_refs)
        self.assertTrue(report.affected_relation_ids)
        impact = report.query_impacts[0]
        self.assertTrue(impact.depends_on_changed_spec)
        self.assertEqual(impact.status, PathStatus.CONFIRMED)

        # a proof that never cited the changed spec stays valid as-is
        other = CallBehaviorRegistry(specs=report.result.call_behavior_specs).declare(
            CallBehaviorSpec("b-note", 1, callee_address=9, may_return=True)
        )
        second = analyzer.revise_call_behavior(report.result, other, queries=(impact.derived_query,))
        self.assertFalse(second.query_impacts[0].depends_on_changed_spec)
        self.assertEqual(second.query_impacts[0].status, PathStatus.CONFIRMED)

    def test_withdraw_returns_to_insufficient_not_unreachable(self):
        analyzer = ControlFlowAnalyzer(make_space(self.unknown_callee_program()))
        first = analyzer.analyze(0)
        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec("sub-return", 1, callee_address=4, may_return=True)
        )
        declared = analyzer.revise_call_behavior(first, registry)
        confirmed = declared.result.paths_to(3)
        self.assertEqual(confirmed.status, PathStatus.CONFIRMED)

        withdrawn = analyzer.withdraw_call_behavior(
            declared.result, "sub-return", queries=(confirmed,)
        )

        query = withdrawn.result.paths_to(3)
        self.assertEqual(query.status, PathStatus.INSUFFICIENT)
        self.assertNotEqual(query.status, PathStatus.UNREACHABLE)
        self.assertTrue(any(item.kind == "procedure_return" for item in query.missing_behaviors))
        self.assertEqual(withdrawn.query_impacts[0].previous_status, PathStatus.CONFIRMED)
        self.assertEqual(withdrawn.query_impacts[0].status, PathStatus.INSUFFICIENT)
        self.assertIn("sub-return", withdrawn.result.revisions[-1].detail)

    def test_conflicting_specs_show_conflict_scope_and_block_paths(self):
        registry = (
            CallBehaviorRegistry()
            .declare(CallBehaviorSpec("optimistic", 1, callee_address=4, may_return=True))
            .declare(CallBehaviorSpec("pessimistic", 1, callee_address=4, may_return=False))
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.unknown_callee_program()), call_behaviors=registry
        )
        result = analyzer.analyze(0)

        self.assertEqual(len(result.behavior_conflicts), 1)
        conflict = result.behavior_conflicts[0]
        self.assertEqual(set(conflict.spec_refs), {"optimistic@v1", "pessimistic@v1"})
        relationship = result.call_relationships[0]
        self.assertIsNone(relationship.callee_may_return)

        query = result.paths_to(3)
        self.assertEqual(query.status, PathStatus.CONFLICT)
        self.assertEqual(query.paths, ())
        self.assertIn("optimistic@v1", query.conflicts[0].constraint_ids)
        self.assertIn("pessimistic@v1", query.conflicts[0].constraint_ids)

    def test_resume_records_exploration_and_assumption_reasons(self):
        data = bytes([0x12, 0x00, 0x08, 0x10, 0x00, 0x06, 0x01, 0x00, 0x11, 0x00])
        analyzer = ControlFlowAnalyzer(make_space(data))
        first = analyzer.analyze(0, budget=AnalysisBudget(max_instructions=2))
        self.assertTrue(first.budget_exhausted)

        resumed = analyzer.resume(first, budget=AnalysisBudget(max_instructions=10))
        kinds = [record.kind for record in resumed.continuations]
        self.assertEqual(kinds, [ContinuationKind.EXPLORATION])
        self.assertEqual(resumed.continuations[0].budget_max_instructions, 10)

        again = analyzer.resume(
            resumed,
            added_specs=(CallBehaviorSpec("sub-return", 1, callee_address=8, may_return=True),),
        )
        kinds = [record.kind for record in again.continuations]
        self.assertEqual(kinds, [ContinuationKind.EXPLORATION, ContinuationKind.ASSUMPTION_CHANGE])
        self.assertEqual(again.continuations[1].spec_refs, ("sub-return@v1",))
        self.assertEqual(len(resumed.continuations), 1)

    def test_compare_behavior_sets_on_mixed_program(self):
        program = recursive_tail_table()
        symbols = program.symbols
        space = make_space(program.data)
        analyzer = ControlFlowAnalyzer(
            space,
            jump_tables=[JumpTableConstraint(symbols["table"], registers=(1,), entry_count=3, max_entries=3)],
        )
        first = analyzer.analyze(symbols["entry"])

        returning = CallBehaviorRegistry().declare(
            CallBehaviorSpec("recur-returns", 1, callee_address=symbols["recur"], may_return=True)
        )
        non_returning = CallBehaviorRegistry().declare(
            CallBehaviorSpec("recur-returns", 1, callee_address=symbols["recur"], may_return=False)
        )
        comparison = analyzer.compare_call_behaviors(
            first, symbols["after"], returning, non_returning
        )

        self.assertEqual(comparison.left.status, PathStatus.ASSUMED)
        self.assertIn("recur-returns@v1", comparison.left.behavior_refs)
        self.assertNotEqual(comparison.left.status, comparison.right.status)
        self.assertTrue(comparison.differences)
        self.assertEqual(comparison.left_spec_refs, ("recur-returns@v1",))
        # the original result is unchanged by the comparison
        self.assertEqual(first.call_behavior_specs, ())

    def test_behavior_state_survives_json_roundtrip(self):
        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec("sub-return", 2, callee_address=4, may_return=True)
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.unknown_callee_program()), call_behaviors=registry
        )
        result = analyzer.analyze(0)
        reloaded = AnalysisResult.from_json(result.to_json())

        self.assertEqual(reloaded.call_behavior_specs, result.call_behavior_specs)
        self.assertEqual(reloaded.call_behavior_specs[0].ref, "sub-return@v2")
        self.assertEqual(
            reloaded.call_relationships[0].behavior_refs,
            result.call_relationships[0].behavior_refs,
        )
        query = reloaded.paths_to(3)
        self.assertIn("sub-return@v2", query.behavior_refs)


class ConditionalCallBehaviorTests(unittest.TestCase):
    def input_call_program(self):
        # call sub (0) ; halt (3) ; sub: jmpr r0 (4)
        return bytes([0x12, 0x00, 0x04, 0x01, 0x11, 0x00])

    def two_site_program(self):
        # 0: call 4 (a) ; 3: halt ; 4: call 9 (sub) ; 7: ret ;
        # 8: call 9 (direct site, second root) ; 11: jmpr r0 ; 13: halt
        return bytes([
            0x12, 0x00, 0x04,
            0x01,
            0x12, 0x00, 0x09,
            0x13,
            0x12, 0x00, 0x09,
            0x11, 0x00,
            0x01,
        ])

    def test_region_scoped_spec_is_not_generalized_to_whole_function(self):
        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec(
                "sub-returns-small",
                1,
                callee_address=4,
                may_return=True,
                applicability=SpecApplicability(entry_values=((0, 0, 9),)),
            )
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.input_call_program()), call_behaviors=registry
        )
        result = analyzer.analyze(0)

        # statically the return stays unknown: the region-scoped note is
        # not generalized to the whole function
        self.assertIsNone(result.call_relationships[0].callee_may_return)

        unconstrained = result.paths_to(3)
        self.assertEqual(unconstrained.status, PathStatus.INSUFFICIENT)
        self.assertTrue(
            any(item.kind == "call_behavior_condition" for item in unconstrained.missing_behaviors)
        )

        # same call, input inside the declared region: conditional conclusion
        inside = result.paths_to(
            3, entry_values=(EntryValueConstraint("ev-in", 0, 0, 5),)
        )
        self.assertEqual(inside.status, PathStatus.ASSUMED)
        self.assertIn("sub-returns-small@v1", inside.behavior_refs)
        self.assertTrue(inside.conditional_paths)
        self.assertFalse(inside.direct_paths)
        self.assertEqual(inside.conditional_paths[0].basis, "conditional")

        # same call, input outside the region: conditions insufficient
        outside = result.paths_to(
            3, entry_values=(EntryValueConstraint("ev-out", 0, 0, 50),)
        )
        self.assertEqual(outside.status, PathStatus.INSUFFICIENT)
        self.assertNotIn("sub-returns-small@v1", outside.behavior_refs)
        self.assertTrue(
            any(item.kind == "call_behavior_condition" for item in outside.missing_behaviors)
        )

    def test_direct_proof_has_direct_basis(self):
        result = ControlFlowAnalyzer(make_space(branch_loop().data)).analyze(0)
        query = result.paths_to(0)
        self.assertEqual(query.status, PathStatus.CONFIRMED)
        self.assertTrue(query.direct_paths)
        self.assertFalse(query.conditional_paths)
        self.assertEqual(query.direct_paths[0].basis, "direct")

    def test_local_conflict_only_blocks_queries_in_the_overlap(self):
        registry = (
            CallBehaviorRegistry()
            .declare(
                CallBehaviorSpec(
                    "small-returns",
                    1,
                    callee_address=4,
                    may_return=True,
                    applicability=SpecApplicability(entry_values=((0, 0, 9),)),
                )
            )
            .declare(
                CallBehaviorSpec(
                    "mid-noreturn",
                    1,
                    callee_address=4,
                    may_return=False,
                    applicability=SpecApplicability(entry_values=((0, 5, 15),)),
                )
            )
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.input_call_program()), call_behaviors=registry
        )
        result = analyzer.analyze(0)

        # the conflict is kept local: only the contested r0 in [5, 9] region
        self.assertEqual(len(result.behavior_conflicts), 1)
        conflict = result.behavior_conflicts[0]
        self.assertEqual(conflict.entry_values, ((0, 5, 9),))
        self.assertEqual(set(conflict.spec_refs), {"small-returns@v1", "mid-noreturn@v1"})

        contested = result.paths_to(3, entry_values=(EntryValueConstraint("ev-c", 0, 0, 7),))
        self.assertEqual(contested.status, PathStatus.CONFLICT)
        self.assertEqual(contested.paths, ())

        # outside the overlap each spec still applies on its own
        left = result.paths_to(3, entry_values=(EntryValueConstraint("ev-l", 0, 0, 3),))
        self.assertEqual(left.status, PathStatus.ASSUMED)
        self.assertEqual(left.behavior_refs, ("small-returns@v1",))

        right = result.paths_to(3, entry_values=(EntryValueConstraint("ev-r", 0, 0, 12),))
        self.assertNotEqual(right.status, PathStatus.CONFLICT)
        self.assertEqual(right.conflicts, ())
        self.assertEqual(right.paths, ())
        self.assertTrue(
            any("call behavior spec" in item.reason for item in right.excluded_relations)
        )

    def test_disjoint_condition_regions_do_not_conflict(self):
        registry = (
            CallBehaviorRegistry()
            .declare(
                CallBehaviorSpec(
                    "small-returns",
                    1,
                    callee_address=4,
                    may_return=True,
                    applicability=SpecApplicability(entry_values=((0, 0, 9),)),
                )
            )
            .declare(
                CallBehaviorSpec(
                    "large-noreturn",
                    1,
                    callee_address=4,
                    may_return=False,
                    applicability=SpecApplicability(entry_values=((0, 10, 99),)),
                )
            )
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.input_call_program()), call_behaviors=registry
        )
        result = analyzer.analyze(0)
        self.assertEqual(result.behavior_conflicts, ())
        query = result.paths_to(3, entry_values=(EntryValueConstraint("ev", 0, 0, 4),))
        self.assertEqual(query.status, PathStatus.ASSUMED)
        self.assertEqual(query.behavior_refs, ("small-returns@v1",))

    def test_call_chain_condition_applies_only_along_that_chain(self):
        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec(
                "sub-returns-via-a",
                1,
                callee_address=9,
                may_return=True,
                applicability=SpecApplicability(call_chain=(4,)),
            )
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.two_site_program()), call_behaviors=registry
        )
        result = analyzer.analyze(0, 8)

        # reachable through the chain entry -> a -> sub -> ret -> halt
        via_chain = result.paths_to(7, source_address=0)
        self.assertEqual(via_chain.status, PathStatus.ASSUMED)
        self.assertIn("sub-returns-via-a@v1", via_chain.behavior_refs)
        for path in via_chain.paths:
            returns = [
                step.source_address for step in path.steps if step.kind is EdgeKind.CALL_RETURN
            ]
            self.assertIn(4, returns)

        # the direct call site at 0 is not on the declared chain: the spec
        # does not apply there and the return stays unproven
        direct = result.paths_to(11, source_address=8)
        self.assertEqual(direct.status, PathStatus.INSUFFICIENT)
        self.assertTrue(
            any(item.kind == "call_behavior_condition" for item in direct.missing_behaviors)
        )

    def test_revision_rejudges_applicability_and_old_query_keeps_version(self):
        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec(
                "sub-returns-small",
                1,
                callee_address=4,
                may_return=True,
                applicability=SpecApplicability(entry_values=((0, 0, 9),)),
            )
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.input_call_program()), call_behaviors=registry
        )
        first = analyzer.analyze(0)
        old_query = first.paths_to(3, entry_values=(EntryValueConstraint("ev", 0, 0, 5),))
        self.assertEqual(old_query.behavior_refs, ("sub-returns-small@v1",))

        widened, _ = registry.revise(
            "sub-returns-small",
            applicability=SpecApplicability(entry_values=((0, 0, 19),)),
        )
        report = analyzer.revise_call_behavior(first, widened, queries=(old_query,))

        # new queries re-judge applicability against the new version
        grown = report.result.paths_to(3, entry_values=(EntryValueConstraint("ev2", 0, 0, 15),))
        self.assertEqual(grown.status, PathStatus.ASSUMED)
        self.assertEqual(grown.behavior_refs, ("sub-returns-small@v2",))
        # the old query object still cites the version it used
        self.assertEqual(old_query.behavior_refs, ("sub-returns-small@v1",))
        self.assertTrue(report.query_impacts[0].depends_on_changed_spec)

        withdrawn = analyzer.withdraw_call_behavior(report.result, "sub-returns-small")
        rejudged = withdrawn.result.paths_to(
            3, entry_values=(EntryValueConstraint("ev3", 0, 0, 5),)
        )
        self.assertEqual(rejudged.status, PathStatus.INSUFFICIENT)

    def test_resume_cannot_widen_applicability_conditions(self):
        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec(
                "sub-returns-small",
                1,
                callee_address=4,
                may_return=True,
                applicability=SpecApplicability(entry_values=((0, 0, 9),)),
            )
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.input_call_program()), call_behaviors=registry
        )
        first = analyzer.analyze(0, budget=AnalysisBudget(max_instructions=2))
        self.assertTrue(first.budget_exhausted)

        with self.assertRaises(ValueError):
            analyzer.resume(
                first,
                added_specs=(
                    CallBehaviorSpec(
                        "sub-returns-small",
                        1,
                        callee_address=4,
                        may_return=True,
                        applicability=SpecApplicability(entry_values=((0, 0, 99),)),
                    ),
                ),
            )

        narrowed = analyzer.resume(
            first,
            added_specs=(
                CallBehaviorSpec(
                    "sub-returns-small",
                    1,
                    callee_address=4,
                    may_return=True,
                    applicability=SpecApplicability(entry_values=((0, 0, 4),)),
                ),
            ),
        )
        self.assertEqual(
            narrowed.continuations[-1].kind, ContinuationKind.ASSUMPTION_CHANGE
        )
        inside = narrowed.paths_to(3, entry_values=(EntryValueConstraint("ev", 0, 0, 2),))
        self.assertEqual(inside.behavior_refs, ("sub-returns-small@v2",))
        outside = narrowed.paths_to(3, entry_values=(EntryValueConstraint("ev2", 0, 0, 7),))
        self.assertEqual(outside.status, PathStatus.INSUFFICIENT)

    def test_conditional_spec_state_survives_json_roundtrip(self):
        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec(
                "sub-returns-small",
                1,
                callee_address=4,
                may_return=True,
                applicability=SpecApplicability(entry_values=((0, 0, 9),), call_chain=(0,)),
            )
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.input_call_program()), call_behaviors=registry
        )
        result = analyzer.analyze(0)
        reloaded = AnalysisResult.from_json(result.to_json())

        spec = reloaded.call_behavior_specs[0]
        self.assertEqual(spec.applicability.entry_values, ((0, 0, 9),))
        self.assertEqual(spec.applicability.call_chain, (0,))
        query = reloaded.paths_to(3, entry_values=(EntryValueConstraint("ev", 0, 0, 5),))
        self.assertIn("sub-returns-small@v1", query.behavior_refs)

    def test_input_dependent_call_sample(self):
        program = input_dependent_call()
        result = ControlFlowAnalyzer(make_space(program.data)).analyze(program.symbols["entry"])
        self.assertIsNone(result.call_relationships[0].callee_may_return)


class PartitionReportTests(unittest.TestCase):
    def input_call_program(self):
        # call sub (0) ; halt (3) ; sub: jmpr r0 (4)
        return bytes([0x12, 0x00, 0x04, 0x01, 0x11, 0x00])

    def analyze_with(self, registry):
        analyzer = ControlFlowAnalyzer(
            make_space(self.input_call_program()), call_behaviors=registry
        )
        return analyzer.analyze(0)

    def assert_complete_partition(self, report, low, high):
        # no gaps, no overlaps, exact coverage of [low, high]
        self.assertEqual(report.low, low)
        self.assertEqual(report.high, high)
        cursor = low
        for segment in report.segments:
            self.assertEqual(segment.low, cursor)
            self.assertLessEqual(segment.low, segment.high)
            cursor = segment.high + 1
        self.assertEqual(cursor, high + 1)

    def test_partition_with_negative_and_adjacent_endpoints(self):
        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec(
                "sub-returns-small",
                1,
                callee_address=4,
                may_return=True,
                applicability=SpecApplicability(entry_values=((0, 0, 9),)),
            )
        )
        result = self.analyze_with(registry)
        report = result.partition_report(3, 0, -5, 20)

        self.assert_complete_partition(report, -5, 20)
        categories = [segment.category for segment in report.segments]
        self.assertEqual(
            categories,
            [
                PartitionCategory.INSUFFICIENT,
                PartitionCategory.CONDITIONAL,
                PartitionCategory.INSUFFICIENT,
            ],
        )
        bounds = [(segment.low, segment.high) for segment in report.segments]
        self.assertEqual(bounds, [(-5, -1), (0, 9), (10, 20)])
        conditional = report.segments[1]
        self.assertEqual(conditional.behavior_refs, ("sub-returns-small@v1",))
        self.assertEqual(report.segments[0].behavior_refs, ())
        self.assertEqual(report.spec_refs, ("sub-returns-small@v1",))

    def test_partition_marks_local_conflict_and_spec_based_unreachable(self):
        registry = (
            CallBehaviorRegistry()
            .declare(
                CallBehaviorSpec(
                    "small-returns",
                    1,
                    callee_address=4,
                    may_return=True,
                    applicability=SpecApplicability(entry_values=((0, 0, 9),)),
                )
            )
            .declare(
                CallBehaviorSpec(
                    "mid-noreturn",
                    1,
                    callee_address=4,
                    may_return=False,
                    applicability=SpecApplicability(entry_values=((0, 5, 15),)),
                )
            )
        )
        result = self.analyze_with(registry)
        report = result.partition_report(3, 0, -2, 20)

        self.assert_complete_partition(report, -2, 20)
        bounds = [(segment.low, segment.high) for segment in report.segments]
        self.assertEqual(bounds, [(-2, -1), (0, 4), (5, 9), (10, 15), (16, 20)])
        categories = [segment.category for segment in report.segments]
        self.assertEqual(
            categories,
            [
                PartitionCategory.INSUFFICIENT,
                PartitionCategory.CONDITIONAL,
                PartitionCategory.CONFLICT,
                PartitionCategory.CONDITIONAL,
                PartitionCategory.INSUFFICIENT,
            ],
        )
        self.assertEqual(report.segments[1].behavior_refs, ("small-returns@v1",))
        # the conflict slice cites exactly the two contradicting versions
        # and the contested condition region, not just the overview list
        self.assertEqual(
            report.segments[2].behavior_refs,
            ("mid-noreturn@v1", "small-returns@v1"),
        )
        self.assertEqual(report.segments[2].conflict_conditions, ((0, 5, 9),))
        self.assertEqual(report.segments[1].conflict_conditions, ())
        # unreachable only because mid-noreturn says so: still evidence-based
        self.assertEqual(report.segments[3].behavior_refs, ("mid-noreturn@v1",))

    def test_partition_conflict_cites_only_participating_specs(self):
        registry = (
            CallBehaviorRegistry()
            .declare(
                CallBehaviorSpec(
                    "a-returns",
                    1,
                    callee_address=4,
                    may_return=True,
                    applicability=SpecApplicability(entry_values=((0, 0, 9),)),
                )
            )
            .declare(
                CallBehaviorSpec(
                    "b-noreturn",
                    1,
                    callee_address=4,
                    may_return=False,
                    applicability=SpecApplicability(entry_values=((0, 3, 12),)),
                )
            )
            .declare(
                CallBehaviorSpec(
                    "c-noreturn",
                    1,
                    callee_address=4,
                    may_return=False,
                    applicability=SpecApplicability(entry_values=((0, 6, 15),)),
                )
            )
            .declare(
                CallBehaviorSpec(
                    "far-returns",
                    1,
                    callee_address=4,
                    may_return=True,
                    applicability=SpecApplicability(entry_values=((0, 100, 200),)),
                )
            )
        )
        result = self.analyze_with(registry)
        report = result.partition_report(3, 0, -2, 20)

        self.assert_complete_partition(report, -2, 20)
        by_bounds = {(segment.low, segment.high): segment for segment in report.segments}
        # a vs b overlap on [3, 9]; a vs c on [6, 9]; b and c agree
        first = by_bounds[(3, 5)]
        self.assertEqual(first.category, PartitionCategory.CONFLICT)
        self.assertEqual(first.behavior_refs, ("a-returns@v1", "b-noreturn@v1"))
        self.assertEqual(first.conflict_conditions, ((0, 3, 9),))
        # three specs overlap here: all participating versions are cited
        second = by_bounds[(6, 9)]
        self.assertEqual(second.category, PartitionCategory.CONFLICT)
        self.assertEqual(
            second.behavior_refs,
            ("a-returns@v1", "b-noreturn@v1", "c-noreturn@v1"),
        )
        self.assertEqual(second.conflict_conditions, ((0, 3, 9), (0, 6, 9)))
        # b and c agree on no-return: conditional, not a conflict
        agreed = by_bounds[(10, 12)]
        self.assertEqual(agreed.category, PartitionCategory.CONDITIONAL)
        self.assertEqual(agreed.behavior_refs, ("b-noreturn@v1", "c-noreturn@v1"))
        # the disjoint far-region spec stays in the overview only
        self.assertIn("far-returns@v1", report.spec_refs)
        for segment in report.segments:
            self.assertNotIn("far-returns@v1", segment.behavior_refs)

    def test_conflict_report_snapshot_survives_revision_and_withdrawal(self):
        registry = (
            CallBehaviorRegistry()
            .declare(
                CallBehaviorSpec(
                    "small-returns",
                    1,
                    callee_address=4,
                    may_return=True,
                    applicability=SpecApplicability(entry_values=((0, 0, 9),)),
                )
            )
            .declare(
                CallBehaviorSpec(
                    "mid-noreturn",
                    1,
                    callee_address=4,
                    may_return=False,
                    applicability=SpecApplicability(entry_values=((0, 5, 15),)),
                )
            )
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.input_call_program()), call_behaviors=registry
        )
        first = analyzer.analyze(0)
        report = first.partition_report(3, 0, 0, 20)
        conflict = next(
            segment
            for segment in report.segments
            if segment.category is PartitionCategory.CONFLICT
        )
        self.assertEqual(
            conflict.behavior_refs, ("mid-noreturn@v1", "small-returns@v1")
        )
        exported = report.to_json()

        revised_registry, _ = registry.revise(
            "mid-noreturn",
            applicability=SpecApplicability(entry_values=((0, 5, 6),)),
        )
        revision = analyzer.revise_call_behavior(first, revised_registry)
        withdrawn = analyzer.withdraw_call_behavior(revision.result, "mid-noreturn")

        # the reloaded snapshot still finds the original versions per segment
        reloaded = PartitionReport.from_json(exported)
        self.assertEqual(reloaded, report)
        reloaded_conflict = next(
            segment
            for segment in reloaded.segments
            if segment.category is PartitionCategory.CONFLICT
        )
        self.assertEqual(reloaded_conflict.low, conflict.low)
        self.assertEqual(reloaded_conflict.high, conflict.high)
        self.assertEqual(
            reloaded_conflict.behavior_refs, ("mid-noreturn@v1", "small-returns@v1")
        )
        self.assertEqual(reloaded_conflict.conflict_conditions, ((0, 5, 9),))

        # new reports reflect the revised and withdrawn specs instead
        narrowed = revision.result.partition_report(3, 0, 0, 20)
        narrowed_conflict = next(
            segment
            for segment in narrowed.segments
            if segment.category is PartitionCategory.CONFLICT
        )
        self.assertEqual((narrowed_conflict.low, narrowed_conflict.high), (5, 6))
        self.assertIn("mid-noreturn@v2", narrowed_conflict.behavior_refs)
        after_withdraw = withdrawn.result.partition_report(3, 0, 0, 20)
        self.assertTrue(
            all(
                segment.category is not PartitionCategory.CONFLICT
                for segment in after_withdraw.segments
            )
        )

    def test_partition_direct_proof_needs_no_evidence(self):
        result = ControlFlowAnalyzer(make_space(branch_loop().data)).analyze(0)
        report = result.partition_report(0, 0, -3, 4)
        self.assert_complete_partition(report, -3, 4)
        self.assertEqual(len(report.segments), 1)
        segment = report.segments[0]
        self.assertEqual(segment.category, PartitionCategory.DIRECT)
        self.assertEqual(segment.behavior_refs, ())
        self.assertEqual(report.spec_refs, ())

    def test_exported_report_is_a_frozen_snapshot(self):
        registry = CallBehaviorRegistry().declare(
            CallBehaviorSpec(
                "sub-returns-small",
                1,
                callee_address=4,
                may_return=True,
                applicability=SpecApplicability(entry_values=((0, 0, 9),)),
            )
        )
        analyzer = ControlFlowAnalyzer(
            make_space(self.input_call_program()), call_behaviors=registry
        )
        first = analyzer.analyze(0)
        report = first.partition_report(3, 0, 0, 19)
        self.assertEqual(
            [(segment.low, segment.high) for segment in report.segments],
            [(0, 9), (10, 19)],
        )
        exported = report.to_json()

        widened, _ = registry.revise(
            "sub-returns-small",
            applicability=SpecApplicability(entry_values=((0, 0, 19),)),
        )
        revision = analyzer.revise_call_behavior(first, widened)
        withdrawn = analyzer.withdraw_call_behavior(revision.result, "sub-returns-small")

        # the exported snapshot still cites the version it was built from
        reloaded = PartitionReport.from_json(exported)
        self.assertEqual(reloaded, report)
        self.assertEqual(reloaded.spec_refs, ("sub-returns-small@v1",))
        self.assertEqual(reloaded.segments[0].behavior_refs, ("sub-returns-small@v1",))

        # revisions and withdrawals only shape newly generated reports
        grown = revision.result.partition_report(3, 0, 0, 19)
        self.assertEqual(
            [(segment.low, segment.high) for segment in grown.segments],
            [(0, 19)],
        )
        self.assertEqual(grown.segments[0].behavior_refs, ("sub-returns-small@v2",))
        self.assertNotEqual(grown.report_id, report.report_id)

        after_withdraw = withdrawn.result.partition_report(3, 0, 0, 19)
        self.assertEqual(len(after_withdraw.segments), 1)
        self.assertEqual(
            after_withdraw.segments[0].category, PartitionCategory.INSUFFICIENT
        )
        self.assertEqual(after_withdraw.spec_refs, ())


if __name__ == "__main__":
    unittest.main()
