# teachbin — 教学指令集二进制分析工作库（v1）

针对项目自定义的一套有限教学指令集 **teach-isa v1**，做反汇编与控制流
恢复。不加载、不解析、不运行任何真实可执行格式（ELF/PE 等）；所有样本
由仓库内置的汇编器合成。

## 分层与稳定身份

“文件片段 / 地址空间 / 分析结论”严格分层，身份在层间稳定传递，后续加入
数据引用分析时可直接沿用：

| 层 | 模块 | 职责 | 稳定身份 |
| --- | --- | --- | --- |
| 存储 | `src/teachbin/storage.py` | `FileFragment`（字节片段）与 `Segment`（片段到虚拟地址的映射）、`AddressSpace`（互不重叠的段集合） | `fragment_id`、`segment_id` |
| 指令集 | `src/teachbin/isa.py` | teach-isa v1 指令表与**单条**指令纯解码 | `OpcodeSpec.opcode` |
| 反汇编 | `src/teachbin/disasm.py` | `DisassemblyDB`：指令实体、字节覆盖、重叠冲突、状态分类 | 指令起始地址 + 片段身份 |
| 控制流 | `src/teachbin/cfg.py` | 工作队列式递归下降恢复：基本块、边、未解析边 | `block_id`、`edge_id` |
| 样本 | `src/teachbin/asm.py`、`samples/` | 两遍汇编器与合成样本，仅产生教学字节 | — |

地址与字节溯源：`AddressSpace.provenance(addr) -> (FileFragment, file_offset)`。
同一片段可映射到多个地址（得到不同 `segment_id`），跨段/越界读取返回
`None`，绝不靠从头逐字节猜测。

## teach-isa v1

变长、小端。寄存器 `A/B/C/D`（0..3），相对偏移为有符号 8 位，
目标 = 指令首址 + 指令长度 + rel。

`HALT NOP LOADI ADD SUB CMP MOV STORE LOAD JMP JZ JNZ JGT JMPIND CALL RET`

控制流语义由 `FlowKind` 显式区分：`FALL / JUMP / COND / CALL / INDIRECT /
RETURN / HALT`。

## 明确保留的状态

分析器**不做线性扫描假设**，任意地址都可能是解码起点（例如来自跳转目标）：

- **重叠解码**：不同起点对同一字节解码时，`overlap_conflicts()` 逐字节
  列出所有主张者；字节分类为 `CODE_OVERLAP`，旧结论不被覆盖。
- **截断指令**：操作码有效但连续可读字节不足，状态为 `TRUNCATED`（含
  `expected_length`）；未映射地址产生长度 0 的截断实体。
- **未知操作码**：状态 `UNKNOWN_OPCODE`，只消耗 1 字节，不臆测长度。
- **未分析区域**：`unanalyzed_ranges()` 给出“已映射但从未覆盖”的区间；
  跳转目标落入其中时由工作队列补分析，无需从头猜。
- **不确定的后继**：`JMPIND`（目标在寄存器中）与 `RET`（返回地址动态）
  登记为 `UnresolvedEdge(INDIRECT_JUMP / DYNAMIC_RETURN)`；跳到未映射或
  落在坏指令上登记 `UNMAPPED_TARGET / BAD_TARGET`，均不伪造边。

控制流探测使用非侵入式的 `DisassemblyDB.peek()`，只有沿控制流真正到达的
起点才经 `decode_at()` 落库，避免“探测”凭空制造重叠实体。

## 自带样本

- `loop_sum`：1..10 求和，含条件退出与循环回边，另带独立数据段（STORE→mem[0]）。
- `branch_call`：if/else 两路汇入 + `CALL helper` / `RET`。
- `unresolved`：`JMPIND` + 跳到未映射区域。
- `truncated`：段内只剩 LOADI 操作码，操作数截断。

生成字节副本供人工核对（分析本身不依赖这些文件）：

```bash
PYTHONPATH=src python -m samples.build   # 写入 samples/bin/*.tb
```

## 运行测试

```bash
PYTHONPATH=src:. python -m unittest discover -s tests
```

15 个用例覆盖：片段/段身份与重叠映射、相对目标、标签编码、重叠解码、
截断与未知操作码、循环回边/条件边/基本块、CALL/RET、间接跳转与未映射
目标、以及“目标先落在未分析区域再被工作队列发现”。

## 最小用法

```python
from samples.build import build_loop_sum
from teachbin import AddressSpace, CFAnalyzer, DisassemblyDB

sample = build_loop_sum()
space = AddressSpace("demo")
sample.install(space)
db = DisassemblyDB(space)
cfg = CFAnalyzer(db)
cfg.analyze(sample.entry)
print(cfg.summary())
print(db.listing())
```

## 为数据引用分析预留

- `DecodedInstruction.data_reference()` 已给出 LOAD/STORE 的数据地址；
- 数据地址经 `AddressSpace.provenance()` 解析到具体片段与偏移；
- 字节级分类 `ByteKind` 与覆盖关系表可直接扩展出“数据/代码身份冲突”。
