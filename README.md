# binwork

`binwork` 是第一版二进制分析工作库，只针对项目定义的有限教学指令集。它不加载、不运行真实可执行程序；样本由库内汇编器生成，用于核对反汇编和控制流恢复。

## 身份分层

- `Fragment`：文件片段身份，保存 `fragment_id` 与原始字节。
- `FragmentMapping` / `AddressSpace`：把同一个片段映射到一个或多个虚拟地址区间，并记录每个地址的片段偏移。
- `Instruction`：一次解码结论，具有独立 `instruction_id`；重叠解码不会覆盖已有结论。
- `CfgEdge` / `BasicBlock`：控制流边和基本块，分别有独立 `edge_id`、`block_id`；边保存 `state`、`basis` 与 `evidence`。
- `IndirectCandidate`：受限间接跳转候选，只携带证据和候选状态，不会自动成为 CFG 边。
- `ProcedureInfo`：跨过程固定点得到的过程入口和 `may_return=True/False/None` 摘要。

这种分层让后续数据引用分析可以继续引用片段、地址位置或具体指令，而不需要把字节来源和分析结论绑定在一起。

## 恢复规则

- 从入口或显式代码根开始递归下降；未被控制流或根指向的地址不会被从头逐字节扫描。
- 直接跳转目标可以位于尚未分析区域，先记录为 `queued`，到达后改为 `resolved`。
- 未映射目标记录为 `unmapped`，对应边保留为 unresolved，不抛异常。
- `JMPR Rs` 是寄存器间接跳转；默认记录 unresolved indirect jump，不猜测唯一目标。
- 可通过 `known_constants` 或人工指定的 `JumpTable` 提供候选。候选目标记录证据并保持 `candidate`，不生成确定 CFG 边；截断表会在 unresolved 边的 evidence 中记录未读完的 entry。
- 直接 `CALL` 会生成确定调用边，并在过程固定点完成后生成 `call_return` 边：会返回过程为 `confirmed`，不返回过程不生成返回边，信息不足时保留 `candidate`。
- 指向已知过程入口的无条件跳转标为 `tail_call`；递归和尾递归在单调固定点中处理。
- `AnalysisBudget(max_instructions=...)` 限制解码数量。预算用尽时返回 `analysis_state=budget_exhausted` 和已确认部分，不抛异常。
- `binwork.serialization` 可保存/载入 JSON 结果；载入不会重新分析，也不会把 unresolved 或 candidate 自动升级为 confirmed。
- 多字节指令落在片段末尾时保留已读字节并标记 `truncated`。
- 未知操作码保留一字节和 `unknown_opcode`，不生成伪造的顺序边。
- 不同代码根可解码重叠字节；重叠的 `Instruction` 身份都保留，涉及的基本块标记 `overlap=True`。

## 内置样本

- `samples/branch_loop.tbin`：条件分支和计数循环。
- `samples/overlap.tbin`：跳过中间数据区域后到达代码。
- `binwork.samples.indirect_table()`：人工编码的 16 位分支表和未映射表项。
- `binwork.samples.recursive_call()`：递归调用和不返回过程。
- `binwork.samples`：提供生成样本字节和符号表的 Python API。

## 使用

```bash
python -m binwork.cli --sample branch-loop
python -m binwork.cli --sample indirect-table
python -m unittest discover -s tests
```

## 教学 ISA

编码为 8 位操作码加长 1–3 字节的寄存器或大端立即数。完整定义见 `binwork/isa.py`，包括 `NOP`、`HALT`、`LDI`、`ADD`、`SUB`、`SUBI`、`JMP`、`JMPR`、`CALL`、`RET`、`JZ`、`JNZ`、`JL`。
