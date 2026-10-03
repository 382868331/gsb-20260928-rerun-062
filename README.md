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
- `AnalysisResult.paths_to(address, constraints=PathConstraints(...))` 可同时限定入口寄存器值、条件分支选择和调用是否返回；状态分为 `confirmed`、`reachable`、`assumed`、`unreachable`、`insufficient`、`unresolved` 与 `conflict`，每个 `PathStep` 保留边或候选关系身份及其证据。
- 条件冲突时 `PathQueryResult.conflicts` 给出涉及的约束 ID、位置、原因和现有边依据；查询不会返回由互斥分支或互斥调用结果拼出的路径。
- 未知调用返回性或未知间接目标进入 `missing_behaviors`，明确说明还需哪项行为说明；预算停止时 `unfinished_scopes` 保留仍未完成的 queued/unmapped 范围。
- `compare_conditions()` 对同一目标的两组约束做结果对照，逐条列出被包含或排除的关系及原因，而不是只比较路径数量；每个条件查询有稳定 `query_id`，修订后的结果仍保留旧结果、候选和证明身份。
- 截断/部分跳转表只生成带假设身份的候选关系；路径使用候选目标时标记为 `assumed`，不会提升为确认边。
- `PendingTarget` 单独记录预算中止时仍未处理的 queued/unmapped 目标；调用 `resume(previous_result)` 后，已证实关系、候选关系和待处理目标各自保留身份。
- `revise_procedure_return()` 与 `revise_jump_table()` 返回新 `AnalysisResult` 和修订记录，旧结果不被覆盖，可用旧边/候选 ID 比较关系变化。
- `AnalysisBudget(max_instructions=...)` 用尽时返回已确认部分；`AnalysisResult.to_json()` / `from_json()` 可原样重建查询结果。

## 调用行为依据的版本管理

- `CallBehaviorStatement` 是用户声明的一条调用行为依据：调用是否返回（may_return）、返回后哪些寄存器值成立（return_values）、以及返回后续上哪些分支条件适用（branch_choices），可按 call_address 或 callee_address 定位。
- `CallBehaviorBasis` 把若干条依据打包成带 basis_id 与 version 的版本；derive() 派生下一版本并记录 supersedes，withdrawn_statement_ids 标记被撤回的依据，旧版本对象不被修改。
- paths_to(..., basis=...) 的结论带 basis_usage（实际采用的依据版本与用到的 statement ID），每条 ReachablePath 带 basis_statement_ids 与 used_constraint_ids，确认结论可回到当时条件、调用依据和执行路径。
- 同一 basis 内两条依据互相矛盾时，Basis.conflicts() 给出冲突范围，查询状态为 conflict，矛盾双方都不参与证明，不会从各自挑选有利部分拼出路径。
- revise_call_behavior(target, old_basis, new_basis) 返回 CallBehaviorRevision：预览受影响的已确认路径、候选、缺失行为与未完成范围，列出仍有效的 preserved_paths，并给出用新依据派生的 derived_query；旧查询结果原样保留在 previous_query。
- 撤回关键依据后，派生查询回到 insufficient 并在 missing_behaviors 说明缺口，不会把原先可达静默改成不可达。
- compare_call_behaviors(target, left_basis, right_basis) 对同一目标比较两套调用说明，逐条列出只在某套说明下成立的关系及原因。
- ControlFlowAnalyzer(call_behavior=basis) 让依据参与分析本身（不返回的调用不再铺开返回后继）；apply_call_behavior() 记录 call_basis 类修订，resume() 在预算放宽后记录 exploration_budget 类修订，新增探索与假设变更的原因分开保留。

受限解析约束由 `KnownIndirectTarget` 和 `JumpTableConstraint(table_address, registers, entry_count, max_entries, entries)` 提供。调用者必须给出表的精确边界；没有精确边界时即使能读出零散 u16，也不会当成完整表。修订完整表时可传入显式 `entries`。
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
