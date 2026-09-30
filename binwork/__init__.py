"""Binary analysis primitives for the finite teaching instruction set."""

from .address import AddressSpace, FragmentMapping, ReadResult, SourceLocation
from .analysis import (
    AnalysisBudget,
    AnalysisResult,
    AnalysisState,
    BasicBlock,
    CandidateKind,
    CfgEdge,
    ControlFlowAnalyzer,
    EdgeBasis,
    EdgeKind,
    EdgeState,
    IndirectCandidate,
    JumpTable,
    ProcedureInfo,
    TargetStatus,
)
from .fragments import Fragment
from .isa import BranchKind, Instruction, Operation, Operand, OperandKind

__all__ = [
    "AddressSpace",
    "AnalysisResult",
    "AnalysisBudget",
    "AnalysisState",
    "BasicBlock",
    "CandidateKind",
    "CfgEdge",
    "EdgeBasis",
    "EdgeKind",
    "EdgeState",
    "IndirectCandidate",
    "JumpTable",
    "ProcedureInfo",
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
