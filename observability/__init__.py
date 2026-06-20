"""可观测性模块 — 全链路 Tracing + 质量指标埋点.

核心指标：
- 一次命中率（First-attempt success rate）
- 人工介入率（Human intervention rate）
- SQL 执行成功率
- 平均响应时间
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SpanEvent:
    """链路中的一个 Span."""
    span_id: str
    parent_span_id: str | None
    node_name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float = 0.0
    status: str = "ok"  # ok, error, pending
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Tracer:
    """轻量级 Tracing 实现. 生产环境可对接 OpenTelemetry."""

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.spans: list[SpanEvent] = []
        self._span_stack: list[str] = []  # span_id stack
        self.start_time = time.time()

    def start_span(self, node_name: str, metadata: dict[str, Any] | None = None) -> str:
        """开启一个新 Span."""
        parent_id = self._span_stack[-1] if self._span_stack else None
        span = SpanEvent(
            span_id=uuid.uuid4().hex[:12],
            parent_span_id=parent_id,
            node_name=node_name,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self.spans.append(span)
        self._span_stack.append(span.span_id)
        return span.span_id

    def end_span(self, status: str = "ok", error: str | None = None) -> None:
        """结束当前 Span."""
        if not self._span_stack:
            return
        span_id = self._span_stack.pop()
        for span in self.spans:
            if span.span_id == span_id:
                span.end_time = time.time()
                span.duration_ms = (span.end_time - span.start_time) * 1000
                span.status = status
                span.error = error
                break

    def get_summary(self) -> dict[str, Any]:
        """获取 Trace 摘要."""
        total_duration = (time.time() - self.start_time) * 1000
        error_spans = [s for s in self.spans if s.status == "error"]
        return {
            "trace_id": self.trace_id,
            "total_duration_ms": round(total_duration, 2),
            "spans_count": len(self.spans),
            "error_count": len(error_spans),
            "has_errors": len(error_spans) > 0,
            "node_path": [s.node_name for s in self.spans if s.parent_span_id is None or not s.parent_span_id],
        }

    def to_json(self) -> str:
        return json.dumps({
            "trace_id": self.trace_id,
            "spans": [
                {
                    "node": s.node_name,
                    "duration_ms": round(s.duration_ms, 2),
                    "status": s.status,
                    "error": s.error,
                }
                for s in self.spans
            ],
            "summary": self.get_summary(),
        }, ensure_ascii=False)


# ── 质量指标采集器 ──────────────────────────────────────────

@dataclass
class QualityMetrics:
    """质量指标."""
    trace_id: str = ""
    first_attempt_success: bool = False  # 一次命中
    human_intervention: bool = False      # 是否需人工介入
    sql_execution_success: bool = False   # SQL 执行成功
    plan_rejected: bool = False           # 计划被驳回
    response_time_ms: float = 0.0         # 响应时间
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "first_attempt_success": self.first_attempt_success,
            "human_intervention": self.human_intervention,
            "sql_execution_success": self.sql_execution_success,
            "plan_rejected": self.plan_rejected,
            "response_time_ms": round(self.response_time_ms, 2),
            "timestamp": self.timestamp,
        }


class MetricsCollector:
    """质量指标采集器. 生产环境应导出到 Prometheus/ES."""

    def __init__(self) -> None:
        self.metrics: list[QualityMetrics] = []

    def record(self, metric: QualityMetrics) -> None:
        self.metrics.append(metric)

    def get_summary(self) -> dict[str, Any]:
        if not self.metrics:
            return {"total_queries": 0}

        total = len(self.metrics)
        successes = sum(1 for m in self.metrics if m.first_attempt_success)
        sql_ok = sum(1 for m in self.metrics if m.sql_execution_success)
        interventions = sum(1 for m in self.metrics if m.human_intervention)
        avg_time = sum(m.response_time_ms for m in self.metrics) / total if total else 0

        return {
            "total_queries": total,
            "first_attempt_success_rate": round(successes / total * 100, 1),
            "sql_execution_success_rate": round(sql_ok / total * 100, 1),
            "human_intervention_rate": round(interventions / total * 100, 1),
            "avg_response_time_ms": round(avg_time, 2),
        }


# ── 全局实例 ────────────────────────────────────────────────

_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _collector
