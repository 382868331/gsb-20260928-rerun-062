"""Binary analysis primitives for the finite teaching instruction set."""

from .address import AddressSpace, FragmentMapping, ReadResult, SourceLocation
from .analysis import BasicBlock, CfgEdge, ControlFlowAnalyzer, AnalysisResult, EdgeKind, TargetStatus
from .fragments import Fragment
from .isa import BranchKind, Instruction, Operation, Operand, OperandKind

__all__ = [
    "AddressSpace",
    "AnalysisResult",
    "BasicBlock",
    "CfgEdge",
    "EdgeKind",
    "TargetStatus",
    "BranchKind",
    "ControlFlowAnalyzer",
    "Fragment",
    "FragmentMapping",
    "Instruction",
    "Operand",
    "OperandKind",
    "Operation",
    "ReadResult",
    "SourceLocation",
]
