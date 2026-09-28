import unittest

from samples.build import (
    build_branch_call,
    build_loop_sum,
    build_truncated_fragment,
    build_unresolved_sample,
)
from teachbin import (
    ByteKind,
    CFAnalyzer,
    DecodeStatus,
    DisassemblyDB,
    EdgeKind,
    InstructionSet,
    UnresolvedReason,
)
from teachbin.storage import AddressSpace


def analyze_sample(sample, *extra_entries):
    space = AddressSpace(sample.name)
    sample.install(space)
    db = DisassemblyDB(space)
    cfg = CFAnalyzer(db)
    cfg.analyze(sample.entry, *extra_entries)
    return space, db, cfg


class TestDisassemblyStates(unittest.TestCase):
    def test_truncated_fragment_is_reported_not_guessed(self):
        space, db, cfg = analyze_sample(build_truncated_fragment())
        truncated = db.truncated()
        self.assertEqual(len(truncated), 1)
        self.assertEqual(truncated[0].address, 0x4000)
        self.assertEqual(truncated[0].expected_length, 3)
        self.assertTrue(
            any(u.kind is UnresolvedReason.DECODE_ERROR for u in cfg.unresolved)
        )
        self.assertEqual(cfg.blocks, [])

    def test_overlapping_decode_is_recorded_as_conflict(self):
        # 字节 10 00 11 01：从 0 解码是 LOADI A,0x11；从 2 解码是 ADD A。
        # 后者覆盖 2..3，与前者在字节 2、3 上冲突。
        space = AddressSpace()
        frag = space.add_fragment(bytes([0x10, 0x00, 0x11, 0x01]), "raw")
        space.map_segment(frag, 0x100, executable=True)
        db = DisassemblyDB(space)
        first = db.decode_at(0x100)
        second = db.decode_at(0x102)
        self.assertEqual(first.status, DecodeStatus.OK)
        self.assertEqual(second.status, DecodeStatus.OK)
        conflicts = db.overlap_conflicts()
        self.assertEqual([c.address for c in conflicts], [0x102])
        self.assertEqual(db.classify_byte(0x102), ByteKind.CODE_OVERLAP)
        # 0x103 上的 0x01 只属于 ADD 操作数之后的 NOP，不构成冲突
        self.assertEqual(db.classify_byte(0x103), ByteKind.CODE_OK)

    def test_unknown_opcode_status(self):
        space = AddressSpace()
        frag = space.add_fragment(bytes([0x01, 0xEE, 0x01]), "raw")
        space.map_segment(frag, 0x100, executable=True)
        db = DisassemblyDB(space)
        inst = db.decode_at(0x101)
        self.assertEqual(inst.status, DecodeStatus.UNKNOWN_OPCODE)
        self.assertEqual(db.classify_byte(0x101), ByteKind.CODE_UNKNOWN)

    def test_unanalyzed_region_stays_explicit(self):
        sample = build_branch_call()
        space, db, cfg = analyze_sample(sample)
        gaps = db.unanalyzed_ranges()
        # 全部指令都应可由入口（含 CALL 目标）到达
        covered_total = sum(end - start for start, end in db.covered_ranges())
        self.assertEqual(covered_total, len(sample.program.code))
        self.assertEqual(gaps, [])


class TestControlFlowLoop(unittest.TestCase):
    def test_loop_sum_recovery(self):
        sample = build_loop_sum()
        space, db, cfg = analyze_sample(sample)

        labels = {name: sample.vaddr + off
                  for name, off in sample.program.labels.items()}
        loop = labels["loop"]
        exit_addr = labels["exit"]
        body = loop + 4  # CMP(2)+JGT(2) 之后的落空地址

        # 基本块：入口块 / 条件块(CMP+JGT) / 循环体 / 出口块
        starts = sorted(block.start for block in cfg.blocks)
        self.assertEqual(starts, [sample.entry, loop, body, exit_addr])

        edge_pairs = {(e.source, e.target): e.kind for e in cfg.edges}
        # CMP 在 loop，JGT 在 loop+2：taken 边退出，落空边进入循环体
        jgt_addr = loop + 2
        self.assertEqual(edge_pairs[(jgt_addr, exit_addr)], EdgeKind.BRANCH_TAKEN)
        # JMP loop 构成回边
        jmp_addr = exit_addr - 2
        self.assertEqual(edge_pairs[(jmp_addr, loop)], EdgeKind.BRANCH_TAKEN)
        # 条件块落空进入循环体
        self.assertEqual(edge_pairs[(jgt_addr, body)], EdgeKind.FALLTHROUGH)
        # 出口块以 HALT 结束且没有出口边
        exit_block = cfg.block_at(exit_addr)
        self.assertEqual(exit_block.instructions[-1].spec.mnemonic, "HALT")

        # 数据引用仍可解析到独立的数据片段身份
        store = next(i for i in db.ok_instructions() if i.spec.mnemonic == "STORE")
        self.assertEqual(store.data_reference(), 0)
        provenance = space.provenance(0)
        self.assertIsNotNone(provenance)
        self.assertEqual(provenance[0].name, "loop_sum.data@0x0")


class TestControlFlowBranchCall(unittest.TestCase):
    def test_branch_and_call_edges(self):
        sample = build_branch_call()
        space, db, cfg = analyze_sample(sample)
        labels = {name: sample.vaddr + off
                  for name, off in sample.program.labels.items()}

        # then/else 两路都汇入 merge，helper 经 CALL 到达、以 RET 结束
        starts = {block.start for block in cfg.blocks}
        self.assertIn(labels["else"], starts)
        self.assertIn(labels["merge"], starts)
        self.assertIn(labels["helper"], starts)
        self.assertEqual(cfg.call_targets, {labels["helper"]})

        call_addr = labels["merge"]
        edge_pairs = {(e.source, e.target): e.kind for e in cfg.edges}
        self.assertEqual(
            edge_pairs[(call_addr, labels["helper"])], EdgeKind.CALL_EDGE
        )
        # CALL 之后落空到 HALT
        self.assertEqual(
            edge_pairs[(call_addr, call_addr + 2)], EdgeKind.FALLTHROUGH
        )

        ret_inst = db.instructions[labels["helper"] + 5]
        self.assertEqual(ret_inst.spec.mnemonic, "RET")
        reasons = {(u.source, u.kind) for u in cfg.unresolved}
        self.assertIn((ret_inst.address, UnresolvedReason.DYNAMIC_RETURN), reasons)

    def test_block_successors_describe_merge(self):
        sample = build_branch_call()
        _, _, cfg = analyze_sample(sample)
        labels = {name: sample.vaddr + off
                  for name, off in sample.program.labels.items()}
        # then 块中的 JMP merge（JMP 为 2 字节，距 merge 5 字节）
        jmp_then = labels["merge"] - 5
        block = cfg.block_at(jmp_then)
        succ = {target.start for _, target in cfg.successors(block)}
        self.assertEqual(succ, {labels["merge"]})


class TestUnresolved(unittest.TestCase):
    def test_indirect_jump_and_unmapped_target_are_unresolved(self):
        sample = build_unresolved_sample()
        # 0x2005 是 JMP +20：额外作为入口，验证跳到未映射区不被猜测
        space, db, cfg = analyze_sample(sample, 0x2005)
        reasons = {(u.source, u.kind) for u in cfg.unresolved}
        self.assertIn((0x2003, UnresolvedReason.INDIRECT_JUMP), reasons)
        self.assertTrue(
            any(kind is UnresolvedReason.UNMAPPED_TARGET for _, kind in reasons)
        )
        # JMPIND 后面的字节不由它“落空”覆盖，保持为独立的未分析状态
        self.assertEqual(db.classify_byte(0x2005), ByteKind.CODE_OK)  # 由入口2分析

    def test_jump_into_unanalyzed_then_analyzed_region(self):
        # 分支目标先出现在尚未分析区域：加入入口后应被工作队列发现
        sample = build_loop_sum()
        space = AddressSpace("lazy")
        sample.install(space)
        db = DisassemblyDB(space)
        cfg = CFAnalyzer(db)
        labels = {name: sample.vaddr + off
                  for name, off in sample.program.labels.items()}
        self.assertFalse(db.was_decoded(labels["loop"]))
        cfg.analyze(sample.entry)
        self.assertTrue(db.was_decoded(labels["loop"]))
        self.assertIsNotNone(cfg.block_at(labels["loop"]))


if __name__ == "__main__":
    unittest.main()
