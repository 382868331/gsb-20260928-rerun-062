import unittest

from teachbin.asm import assemble
from teachbin.isa import DecodeStatus, FlowKind, InstructionSet
from teachbin.storage import AddressSpace


class TestStorage(unittest.TestCase):
    def test_fragment_and_segment_identities_are_stable(self):
        space = AddressSpace()
        frag_a = space.add_fragment(b"\x01\x02", "a")
        frag_b = space.add_fragment(b"\x03", "b")
        seg_a = space.map_segment(frag_a, 0x100)
        seg_b = space.map_segment(frag_b, 0x200)
        self.assertEqual((frag_a.fragment_id, frag_b.fragment_id), (0, 1))
        self.assertEqual((seg_a.segment_id, seg_b.segment_id), (0, 1))
        provenance = space.provenance(0x101)
        self.assertIsNotNone(provenance)
        frag, off = provenance
        self.assertIs(frag, frag_a)
        self.assertEqual(off, 1)

    def test_segments_must_not_overlap_and_reads_do_not_guess(self):
        space = AddressSpace()
        frag = space.add_fragment(b"\x00" * 4, "a")
        space.map_segment(frag, 0x100)
        with self.assertRaises(ValueError):
            space.map_segment(frag, 0x102)
        self.assertIsNone(space.read(0x102, 10))  # 跨段尾：返回 None
        self.assertEqual(space.read(0x100, 2), b"\x00\x00")
        self.assertFalse(space.is_mapped(0x400))

    def test_same_fragment_can_map_twice_with_distinct_segment_ids(self):
        space = AddressSpace()
        frag = space.add_fragment(b"\x90", "shared")
        s1 = space.map_segment(frag, 0x100)
        s2 = space.map_segment(frag, 0x300)
        self.assertNotEqual(s1.segment_id, s2.segment_id)
        self.assertIs(space.provenance(0x300)[0], frag)


class TestISA(unittest.TestCase):
    def setUp(self):
        self.isa = InstructionSet()

    def test_decode_relative_target_uses_signed_offset(self):
        # JMP -2 at 0x10: target = 0x10 + 2 - 2 = 0x10
        inst = self.isa.decode_at(0x10, bytes([0x30, 0xFE]))
        self.assertEqual(inst.status, DecodeStatus.OK)
        self.assertEqual(inst.flow, FlowKind.JUMP)
        self.assertEqual(inst.direct_target(), 0x10)

    def test_truncated_and_unknown_have_explicit_status(self):
        truncated = self.isa.decode_at(0x20, bytes([0x10]))
        self.assertEqual(truncated.status, DecodeStatus.TRUNCATED)
        self.assertEqual(truncated.expected_length, 3)
        unknown = self.isa.decode_at(0x21, bytes([0xEE]))
        self.assertEqual(unknown.status, DecodeStatus.UNKNOWN_OPCODE)
        empty = self.isa.decode_at(0x22, b"")
        self.assertEqual(empty.status, DecodeStatus.TRUNCATED)

    def test_assembler_encodes_labels_and_data_refs(self):
        program = assemble([
            "start:",
            "LOADI A, 5",
            "JMP start",
            "STORE B, 0x10",
        ])
        self.assertEqual(
            program.code,
            bytes([0x10, 0x00, 0x05, 0x30, 0xFB, 0x20, 0x01, 0x10]),
        )
        inst = self.isa.decode_at(0, program.code[5:8])
        self.assertEqual(inst.data_reference(), 0x10)


if __name__ == "__main__":
    unittest.main()
