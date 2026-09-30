"""Binary analysis primitives for the finite teaching instruction set."""

from .address import AddressSpace, FragmentMapping, ReadResult, SourceLocation
from .analysis import (
    AnalysisBudget,
    AnalysisResult,
    BasicBlock,
    CallRelationship,
    CandidateStatus,
    CfgEdge,
    EdgeEvidence,
    EdgeKind,
    EvidenceKind,
    JumpTableConstraint,
    KnownIndirectTarget,
    Procedure,
    TargetStatus,
    ControlFlowAnalyzer,
)
from .fragments import Fragment
from .isa import BranchKind, Instruction, Operation, Operand, OperandKind

__all__ = [
    "AddressSpace",
    "AnalysisBudget",
    "AnalysisResult",
    "BasicBlock",
    "CallRelationship",
    "CandidateStatus",
    "CfgEdge",
    "EdgeEvidence",
    "EdgeKind",
    "EvidenceKind",
    "JumpTableConstraint",
    "KnownIndirectTarget",
    "Procedure",
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
