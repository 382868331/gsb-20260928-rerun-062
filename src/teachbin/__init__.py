"""teachbin: 面向有限教学指令集的二进制反汇编与控制流恢复工作库。

分层结构：
- ``storage``  文件片段(FileFragment) 与地址空间(AddressSpace)，身份长期稳定；
- ``isa``      教学指令集的静态定义与单条指令解码；
- ``disasm``   反汇编数据库，记录指令、重叠解码、截断与未知状态；
- ``cfg``      基于工作队列的控制流恢复，保留未解析边与明确状态；
- ``asm``      仅供构造样本的简易汇编器，绝不解析真实可执行格式。
"""

from .storage import AddressSpace, FileFragment, Segment
from .isa import DecodedInstruction, InstructionSet, OpcodeSpec, DecodeStatus
from .disasm import DisassemblyDB, ByteKind
from .cfg import (
    BasicBlock,
    CFAnalyzer,
    Edge,
    EdgeKind,
    UnresolvedEdge,
    UnresolvedReason,
)

__all__ = [
    "AddressSpace",
    "FileFragment",
    "Segment",
    "DecodedInstruction",
    "InstructionSet",
    "OpcodeSpec",
    "DecodeStatus",
    "DisassemblyDB",
    "ByteKind",
    "BasicBlock",
    "CFAnalyzer",
    "Edge",
    "EdgeKind",
    "UnresolvedEdge",
    "UnresolvedReason",
]

__version__ = "0.1.0"
