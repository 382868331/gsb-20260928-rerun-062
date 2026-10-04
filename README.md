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
- `CallBehaviorSpec` 是带版本的调用行为依据：声明一次调用可能返回什么（`return_values`）、是否会返回（`may_return`）以及哪些分支条件适用（`branch_choices`）。`CallBehaviorRegistry` 通过 `declare()` / `revise()` / `withdraw()` 管理版本，旧版本进入 `history`，旧注册表不被改写。
- `CallBehaviorSpec.conditions`（`SpecCondition(register, minimum, maximum)`）限定依据的适用输入区域：只有查询约束把对应寄存器定在区间内时依据才支持结论。查询结论用 `PathQueryResult.proof_kind` / `ReachablePath.proof_kind` 区分 `direct`（直接代码证明）、`conditional`（依赖人工条件）与 `insufficient`（条件不足，`missing_behaviors` 给出 `spec_condition` 项说明还需建立哪个寄存器条件）。某个返回值范围下的说明不会被推广到整个函数：落在区域外的查询回到 `insufficient`，被排除的返回边在 `excluded_relations` 中说明原因。
- 两份依据只在条件区域重叠且结论相矛盾时才算冲突：`BehaviorConflict.overlap` 记录重叠区域，区域互不相交的两份依据各自有效。冲突只影响约束落在重叠区域内的查询；区域外的查询各自采用唯一适用的依据并引用其 `spec_id@version`。
- `resume()` 续接递归或预算中断时，同名依据的条件只允许收窄、不允许扩大（否则抛出 `ValueError`）；撤销、替换依据后新查询在新结果上重新判定适用范围，旧查询保留原版本结论与 `spec_id@version` 引用。
- 查询结论通过 `PathQueryResult.behavior_refs`、`ReachablePath.behavior_refs` 和 `PathStep.behavior_refs` 说明实际采用了哪份依据（`spec_id@version`）；由依据决定的返回边带 `call_behavior_spec` 证据。只有过程体自身无法判定返回性时依据才被视为决定性依据并被引用。
- `revise_call_behavior()` / `withdraw_call_behavior()` 返回 `BehaviorImpactReport`：预览受影响的已确认关系、候选与未完成范围，并对传入的旧查询给出 `QueryImpact`。不依赖被改依据的证明继续有效；撤回关键依据后结论回到 `insufficient` 与 `missing_behaviors`，不会自动把可达改成不可达。
- 两份活跃依据互相矛盾时，`AnalysisResult.behavior_conflicts` 展示冲突范围，相关查询状态为 `conflict`，不会从各自挑选有利部分拼出路径。
- `resume(previous, budget=..., added_specs=...)` 用 `ContinuationRecord` 分别记录新增探索（`exploration`）与假设变更（`assumption_change`）的原因；每条确认结论都能回到当时条件、调用依据版本和执行路径。
- `compare_call_behaviors()` 对同一目标比较两套调用说明，逐条列出被包含或排除的关系及原因。
- `AnalysisBudget(max_instructions=...)` 用尽时返回已确认部分；`AnalysisResult.to_json()` / `from_json()` 可原样重建查询结果。

受限解析约束由 `KnownIndirectTarget` 和 `JumpTableConstraint(table_address, registers, entry_count, max_entries, entries)` 提供。调用者必须给出表的精确边界；没有精确边界时即使能读出零散 u16，也不会当成完整表。修订完整表时可传入显式 `entries`。
- 多字节指令落在片段末尾时保留已读字节并标记 `truncated`。
- 未知操作码保留一字节和 `unknown_opcode`，不生成伪造的顺序边。
- 不同代码根可解码重叠字节；重叠的 `Instruction` 身份都保留，涉及的基本块标记 `overlap=True`。

## 内置样本

- `samples/branch_loop.tbin`：条件分支和计数循环。
- `samples/overlap.tbin`：跳过中间数据区域后到达代码。
- `binwork.samples.recursive_tail_table()`：递归调用、尾调用和部分跳转表混合的程序，用于比较两套调用说明。
- `binwork.samples`：提供生成样本字节和符号表的 Python API。

## 使用

```bash
python -m binwork.cli --sample branch-loop
python -m unittest discover -s tests
```

## 教学 ISA

编码为 8 位操作码加长 1–3 字节的寄存器或大端立即数。完整定义见 `binwork/isa.py`，包括 `NOP`、`HALT`、`LDI`、`ADD`、`SUB`、`SUBI`、`JMP`、`JMPR`、`CALL`、`RET`、`JZ`、`JNZ`、`JL`。
