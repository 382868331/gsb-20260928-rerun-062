"""Immutable batch reports for input-value conditional queries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum

from .analysis import (
    AnalysisResult,
    EntryValueConstraint,
    PathQueryResult,
    PathStatus,
)


class PartitionCategory(str, Enum):
    DIRECT = "direct_code"
    MANUAL_EVIDENCE = "manual_evidence"
    LOCAL_CONFLICT = "local_conflict"
    INSUFFICIENT = "insufficient_condition"


@dataclass(frozen=True, slots=True)
class QueryPartition:
    start: int
    end: int
    category: PartitionCategory
    status: PathStatus
    behavior_refs: tuple[str, ...]
    query_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConditionalPartitionReport:
    report_id: str
    target_address: int
    source_address: int
    entry_address: int
    register: int
    low: int
    high: int
    created_at: str
    segments: tuple[QueryPartition, ...]
    analysis: AnalysisResult
    query_snapshots: tuple[dict, ...] = ()

    def covers_input_without_gaps(self) -> bool:
        if not self.segments:
            return False
        if self.segments[0].start != self.low or self.segments[-1].end != self.high:
            return False
        return all(
            current.start == previous.end + 1
            for previous, current in zip(self.segments, self.segments[1:])
        )

    def to_dict(self) -> dict:
        return {
            "schema": "conditional-partition-report/v1",
            "report_id": self.report_id,
            "created_at": self.created_at,
            "query": {
                "target_address": self.target_address,
                "source_address": self.source_address,
                "entry_address": self.entry_address,
                "register": self.register,
                "low": self.low,
                "high": self.high,
            },
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "category": segment.category.value,
                    "status": segment.status.value,
                    "behavior_refs": list(segment.behavior_refs),
                    "query_ids": list(segment.query_ids),
                }
                for segment in self.segments
            ],
            "query_snapshots": list(self.query_snapshots),
            "analysis_snapshot": self.analysis.to_dict(),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "ConditionalPartitionReport":
        query = data["query"]
        return cls(
            report_id=data["report_id"],
            target_address=query["target_address"],
            source_address=query["source_address"],
            entry_address=query["entry_address"],
            register=query["register"],
            low=query["low"],
            high=query["high"],
            created_at=data["created_at"],
            segments=tuple(
                QueryPartition(
                    start=item["start"],
                    end=item["end"],
                    category=PartitionCategory(item["category"]),
                    status=PathStatus(item["status"]),
                    behavior_refs=tuple(item["behavior_refs"]),
                    query_ids=tuple(item["query_ids"]),
                )
                for item in data["segments"]
            ),
            analysis=AnalysisResult.from_dict(data["analysis_snapshot"]),
            query_snapshots=tuple(data.get("query_snapshots", ())),
        )


def partition_input_interval(
    result: AnalysisResult,
    target_address: int,
    low: int,
    high: int,
    *,
    register: int = 0,
    source_address: int | None = None,
    entry_address: int | None = None,
) -> ConditionalPartitionReport:
    """Split every integer in ``[low, high]`` into one proof category.

    Spec applicability endpoints are used as atomic-region boundaries, so
    adjacent ranges and overlaps cannot be skipped or counted twice.
    """

    if low > high:
        raise ValueError("input interval low must not exceed high")
    source = source_address if source_address is not None else result.entry_points[0]
    entry = entry_address if entry_address is not None else source
    boundaries = {low, high + 1}
    for spec in result.call_behavior_specs:
        if spec.applicability.call_chain:
            continue
        for spec_register, spec_low, spec_high in spec.applicability.entry_values:
            if spec_register == register:
                boundaries.update((spec_low, spec_high + 1))

    atomic: list[QueryPartition] = []
    snapshots: list[dict] = []
    for start in sorted(boundaries):
        if start < low or start > high:
            continue
        following = min(point for point in boundaries if point > start)
        end = min(following - 1, high)
        start_query = _query_at(
            result,
            target_address,
            source,
            entry,
            register,
            start,
        )
        end_query = start_query if end == start else _query_at(
            result,
            target_address,
            source,
            entry,
            register,
            end,
        )
        start_category = _category_for(start_query)
        end_category = _category_for(end_query)
        start_refs = _refs_for(start_query)
        end_refs = _refs_for(end_query)
        if (
            start_category is not end_category
            or start_query.status is not end_query.status
            or start_refs != end_refs
        ):
            raise RuntimeError("partition boundaries did not preserve a stable proof category")
        snapshots.append(_query_snapshot(start_query))
        if end_query is not start_query:
            snapshots.append(_query_snapshot(end_query))
        atomic.append(
            QueryPartition(
                start=start,
                end=end,
                category=start_category,
                status=start_query.status,
                behavior_refs=start_refs,
                query_ids=tuple(dict.fromkeys((start_query.query_id, end_query.query_id))),
            )
        )

    merged: list[QueryPartition] = []
    for segment in atomic:
        if (
            merged
            and merged[-1].end + 1 == segment.start
            and merged[-1].category is segment.category
            and merged[-1].status is segment.status
            and merged[-1].behavior_refs == segment.behavior_refs
        ):
            previous = merged[-1]
            merged[-1] = QueryPartition(
                start=previous.start,
                end=segment.end,
                category=previous.category,
                status=previous.status,
                behavior_refs=previous.behavior_refs,
                query_ids=tuple(dict.fromkeys(previous.query_ids + segment.query_ids)),
            )
        else:
            merged.append(segment)

    report_id = _report_id(
        result,
        target_address,
        source,
        entry,
        register,
        low,
        high,
    )
    report = ConditionalPartitionReport(
        report_id=report_id,
        target_address=target_address,
        source_address=source,
        entry_address=entry,
        register=register,
        low=low,
        high=high,
        created_at=datetime.now(timezone.utc).isoformat(),
        segments=tuple(merged),
        analysis=result,
        query_snapshots=tuple(snapshots),
    )
    if not report.covers_input_without_gaps():
        raise RuntimeError("partition report missed or duplicated an input value")
    return report


def _query_at(
    result: AnalysisResult,
    target_address: int,
    source_address: int,
    entry_address: int,
    register: int,
    value: int,
) -> PathQueryResult:
    constraint = EntryValueConstraint(
        f"partition-r{register}:{value}",
        entry_address,
        register,
        value,
    )
    return result.paths_to(
        target_address,
        source_address=source_address,
        entry_values=(constraint,),
        include_statuses=(
            PathStatus.CONFIRMED,
            PathStatus.ASSUMED,
            PathStatus.UNRESOLVED,
            PathStatus.REACHABLE,
            PathStatus.UNREACHABLE,
            PathStatus.INSUFFICIENT,
            PathStatus.CONFLICT,
        ),
    )


def _category_for(query: PathQueryResult) -> PartitionCategory:
    if query.status is PathStatus.CONFLICT:
        return PartitionCategory.LOCAL_CONFLICT
    refs = _refs_for(query)
    if refs and (
        query.status in {PathStatus.ASSUMED, PathStatus.REACHABLE, PathStatus.UNREACHABLE}
        or query.conditional_paths
        or any("call behavior spec" in item.reason for item in query.excluded_relations)
    ):
        return PartitionCategory.MANUAL_EVIDENCE
    if query.status in {PathStatus.CONFIRMED, PathStatus.REACHABLE} and (
        query.direct_paths or not refs
    ):
        return PartitionCategory.DIRECT
    return PartitionCategory.INSUFFICIENT


def _refs_for(query: PathQueryResult) -> tuple[str, ...]:
    if query.behavior_refs:
        return query.behavior_refs
    if query.status is PathStatus.CONFLICT:
        return tuple(
            sorted(
                {
                    ref
                    for conflict in query.conflicts
                    for ref in conflict.constraint_ids
                    if "@v" in ref
                }
            )
        )
    if query.status is PathStatus.UNREACHABLE:
        return query.available_behavior_refs
    if query.available_behavior_refs and any(
        "call behavior spec" in item.reason for item in query.excluded_relations
    ):
        return query.available_behavior_refs
    return ()


def _query_snapshot(query: PathQueryResult) -> dict:
    return {
        "query_id": query.query_id,
        "source_address": query.source_address,
        "target_address": query.target_address,
        "status": query.status.value,
        "constraints": {
            "entry_values": [
                {
                    "constraint_id": item.constraint_id,
                    "entry_address": item.entry_address,
                    "register": item.register,
                    "value": item.value,
                }
                for item in query.constraints.entry_values
            ],
            "branch_choices": [
                {
                    "constraint_id": item.constraint_id,
                    "branch_address": item.branch_address,
                    "taken": item.taken,
                    "target_address": item.target_address,
                }
                for item in query.constraints.branch_choices
            ],
            "call_results": [
                {
                    "constraint_id": item.constraint_id,
                    "may_return": item.may_return,
                    "call_address": item.call_address,
                    "callee_address": item.callee_address,
                }
                for item in query.constraints.call_results
            ],
        },
        "paths": [
            {
                "path_id": path.path_id,
                "status": path.status.value,
                "basis": path.basis,
                "behavior_refs": list(path.behavior_refs),
                "steps": [
                    {
                        "relation_id": step.relation_id,
                        "kind": step.kind.value,
                        "confirmed": step.confirmed,
                        "behavior_refs": list(step.behavior_refs),
                    }
                    for step in path.steps
                ],
            }
            for path in query.paths
        ],
        "conflicts": [
            {
                "conflict_id": item.conflict_id,
                "constraint_ids": list(item.constraint_ids),
                "subject": item.subject,
                "reason": item.reason,
            }
            for item in query.conflicts
        ],
        "missing_behaviors": [
            {
                "requirement_id": item.requirement_id,
                "kind": item.kind,
                "description": item.description,
            }
            for item in query.missing_behaviors
        ],
        "excluded_relations": [
            {
                "relation_id": item.relation_id,
                "reason": item.reason,
                "violated_constraint_ids": list(item.violated_constraint_ids),
            }
            for item in query.excluded_relations
        ],
        "behavior_refs": list(query.behavior_refs),
        "available_behavior_refs": list(query.available_behavior_refs),
    }


def _report_id(
    result: AnalysisResult,
    target_address: int,
    source_address: int,
    entry_address: int,
    register: int,
    low: int,
    high: int,
) -> str:
    payload = json.dumps(
        {
            "target": target_address,
            "source": source_address,
            "entry": entry_address,
            "register": register,
            "low": low,
            "high": high,
            "specs": [
                {
                    "ref": spec.ref,
                    "callee_address": spec.callee_address,
                    "call_address": spec.call_address,
                    "may_return": spec.may_return,
                    "applicability": asdict(spec.applicability),
                }
                for spec in result.call_behavior_specs
            ],
        },
        sort_keys=True,
    ).encode()
    return "partition-report:" + hashlib.sha256(payload).hexdigest()[:16]
