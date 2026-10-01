# binwork

`binwork` 是第一版二进制分析工作库，只针对项目定义的有限教学指令集。它不加载、不运行真实可执行程序；样本由库内汇编器生成，用于核对反汇编和控制流恢复。

## 身份分层

- `Fragment`：文件片段身份，保存 `fragment_id` 与原始字节。
- `FragmentMapping` / `AddressSpace`：把同一个片段映射到一个或多个虚拟地址区间，并记录每个地址的片段偏移。
- `Instruction`：一次解码结论，具有独立 `instruction_id`；重叠解码不会覆盖已有结论。
- `CfgEdge` / `BasicBlock`：控制流边和基本块，分别有独立 `edge_id`、`block_id`。

这种分层让后续数据引用分析可以继续引用片段、地址位置或具体指令，而不需要把字节来源和分析结论绑定在一起。

## 恢复规则

- 从入口或显式代码根开始递归下降；未被控制流或根指向的地址不会被从头逐字节扫描。
- 直接跳转目标可以位于尚未分析区域，先记录为 `queued`，到达后改为 `resolved`。
- 未映射目标记录为 `unmapped`，对应边保留为 unresolved，不抛异常。
- `JMPR Rs` 是寄存器间接跳转；只有明确的已知常量或完整、边界明确的小型跳转表才能生成确定边。
- 截断表或推断出的表项只进入 `indirect_target_candidates`，状态保留为 candidate/truncated；重新载入快照不会把候选提升成确定边。
- `CALL` 同时保留调用边和跨过程返回关系；返回性未知时返回边保持 unresolved，确认被调过程不返回后删除返回后继。
- 指向过程入口的直接尾跳记录为 `tail_call`，不创建静态返回地址。
- 每条 `CfgEdge` 都带 `evidence`，记录直接操作数、已知常量、跳转表、尾调用或返回性分析等依据。
- `AnalysisBudget(max_instructions=...)` 用尽时返回已确认部分；`AnalysisResult.to_json()` / `from_json()` 可原样重建查询结果。

## 可达性查询
`AnalysisResult.reachable_paths()`、`path_to_address()` 和 `path_to_instruction()` 返回带完整 hop/evidence 的路径，并明确分为：
- `confirmed`：所有边和跨过程返回关系都已证实。
- `assumed`：只在调用方显式启用 `assume_candidates=True` 或 `assume_callee_returns=True` 时可达。
- `unresolved`：路径依赖未处理目标、未知间接边或返回性未知的调用；已解码目标仍会保留到未解决 frontier。

预算中止后调用 `ControlFlowAnalyzer.continue_analysis(result)` 会继续处理 `QUEUED` 目标，已有 instruction、edge、candidate 与待处理地址身份保持不变。调用 `revise(result, ...)` 可修订过程返回性、已知间接目标或跳转表边界，并在 `relationship_history` 中保存修订前关系；递归、尾调用和部分跳转表同时存在时，局部确认不会消除未知间接边或未知返回边。

受限解析约束由 `KnownIndirectTarget` 和 `JumpTableConstraint(table_address, registers, entry_count, max_entries)` 提供。调用者必须给出表的精确边界；没有精确边界时即使能读出零散 u16，也不会当成完整表。
- 多字节指令落在片段末尾时保留已读字节并标记 `truncated`。
- 未知操作码保留一字节和 `unknown_opcode`，不生成伪造的顺序边。
- 不同代码根可解码重叠字节；重叠的 `Instruction` 身份都保留，涉及的基本块标记 `overlap=True`。

## 内置样本

- `samples/branch_loop.tbin`：条件分支和计数循环。
- `samples/overlap.tbin`：跳过中间数据区域后到达代码。
- `binwork.samples`：提供生成样本字节和符号表的 Python API。

## 使用

```bash
python -m binwork.cli --sample branch-loop
python -m unittest discover -s tests
```

## 教学 ISA

编码为 8 位操作码加长 1–3 字节的寄存器或大端立即数。完整定义见 `binwork/isa.py`，包括 `NOP`、`HALT`、`LDI`、`ADD`、`SUB`、`SUBI`、`JMP`、`JMPR`、`CALL`、`RET`、`JZ`、`JNZ`、`JL`。
