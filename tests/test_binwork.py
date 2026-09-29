import unittest

from binwork.address import AddressSpace, Fragment, FragmentMapping
from binwork.analysis import EdgeKind, TargetStatus, ControlFlowAnalyzer
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


if __name__ == "__main__":
    unittest.main()
