"""JSON persistence for partial and complete analysis results.

Loading is intentionally not an analysis pass. Candidate and unresolved
edges are restored with the state recorded on disk.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .analysis import (
    AnalysisBudget,
    AnalysisResult,
    AnalysisState,
    BasicBlock,
    CandidateKind,
    CfgEdge,
    EdgeBasis,
    EdgeKind,
    EdgeState,
    IndirectCandidate,
    ProcedureInfo,
    TargetStatus,
)
from .isa import BranchKind, Instruction, Operand, OperandKind, Operation

SCHEMA_VERSION = 1


def _to_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes | bytearray | memoryview):
        return {"__type__": "bytes", "hex": bytes(value).hex()}
    if is_dataclass(value):
        return {item.name: _to_json(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_json(item) for item in value]
    return value


def result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    data = _to_json(result)
    data["schema_version"] = SCHEMA_VERSION
    return data


def save_result(result: AnalysisResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")


def result_from_dict(data: dict[str, Any]) -> AnalysisResult:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported analysis result schema version")

    instructions = tuple(
        Instruction(
            instruction_id=item["instruction_id"],
            address=item["address"],
            operation=Operation(item["operation"]),
            operands=tuple(
                Operand(kind=OperandKind(operand["kind"]), value=operand["value"])
                for operand in item["operands"]
            ),
            encoding=bytes.fromhex(item["encoding"]["hex"]),
            next_address=item["next_address"],
            branch_kind=BranchKind(item["branch_kind"]),
            branch_target=item["branch_target"],
            truncated=item["truncated"],
            unknown_opcode=item["unknown_opcode"],
            source_fragment_id=item["source_fragment_id"],
        )
        for item in data["instructions"]
    )

    edges = tuple(
        CfgEdge(
            edge_id=item["edge_id"],
            source_instruction_id=item["source_instruction_id"],
            kind=EdgeKind(item["kind"]),
            target_address=item["target_address"],
            target_block_id=item["target_block_id"],
            resolved=item["resolved"],
            state=EdgeState(item["state"]),
            basis=EdgeBasis(item["basis"]),
            evidence=item["evidence"],
        )
        for item in data["edges"]
    )
    blocks = tuple(
        BasicBlock(
            block_id=item["block_id"],
            start_address=item["start_address"],
            end_address=item["end_address"],
            instruction_ids=tuple(item["instruction_ids"]),
            outgoing_edge_ids=tuple(item["outgoing_edge_ids"]),
            overlap=item["overlap"],
        )
        for item in data["blocks"]
    )
    candidates = tuple(
        IndirectCandidate(
            candidate_id=item["candidate_id"],
            source_instruction_id=item["source_instruction_id"],
            target_address=item["target_address"],
            kind=CandidateKind(item["kind"]),
            state=EdgeState(item["state"]),
            evidence=item["evidence"],
        )
        for item in data["indirect_candidates"]
    )
    procedures = tuple(
        ProcedureInfo(
            procedure_id=item["procedure_id"],
            entry_address=item["entry_address"],
            may_return=item["may_return"],
            incoming_call_edge_ids=tuple(item["incoming_call_edge_ids"]),
            incoming_tail_call_edge_ids=tuple(item["incoming_tail_call_edge_ids"]),
            evidence=item["evidence"],
        )
        for item in data["procedures"]
    )
    budget_data = data.get("budget") or {}
    budget = AnalysisBudget(max_instructions=budget_data.get("max_instructions"))
    return AnalysisResult(
        entry_points=tuple(data["entry_points"]),
        instructions=instructions,
        edges=edges,
        blocks=blocks,
        target_statuses={int(key): TargetStatus(value) for key, value in data["target_statuses"].items()},
        unresolved_indirect_jumps=tuple(data["unresolved_indirect_jumps"]),
        roots=tuple(data["roots"]),
        indirect_candidates=candidates,
        procedures=procedures,
        analysis_state=AnalysisState(data["analysis_state"]),
        instructions_decoded=data["instructions_decoded"],
        budget=budget,
    )


def load_result(path: str | Path) -> AnalysisResult:
    return result_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
