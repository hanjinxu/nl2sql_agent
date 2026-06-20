"""Data Models for NL2SQL Data Agent — Semantic Layer IR, Plan, State."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── 语义层模型 ──────────────────────────────────────────────

class AggFunction(str, Enum):
    """支持的聚合函数."""
    SUM = "SUM"
    AVG = "AVG"
    COUNT = "COUNT"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    MAX = "MAX"
    MIN = "MIN"


class FieldType(str, Enum):
    DIMENSION = "dimension"  # 维度
    METRIC = "metric"       # 指标


class SchemaField(BaseModel):
    """字段元数据——语义层的原子单元."""
    name: str
    alias: list[str] = Field(default_factory=list)  # 业务别名（"DAU", "日活"）
    type: FieldType
    table: str
    column: str
    data_type: str = "string"  # int, float, string, date
    description: str = ""
    aggregation: AggFunction | None = None
    tags: list[str] = Field(default_factory=list)


class SchemaTable(BaseModel):
    """表元数据."""
    name: str
    alias: list[str] = Field(default_factory=list)
    fields: dict[str, SchemaField] = Field(default_factory=dict)  # name → field
    description: str = ""
    join_keys: dict[str, str] = Field(default_factory=dict)  # table_name → join_key


class MetricDefinition(BaseModel):
    """指标定义——语义层的核心抽象."""
    name: str
    alias: list[str] = Field(default_factory=list)
    description: str = ""
    table: str
    field: str
    aggregation: AggFunction
    filters: list[dict[str, Any]] = Field(default_factory=list)  # pre-defined filters
    tags: list[str] = Field(default_factory=list)


class DimensionDefinition(BaseModel):
    """维度定义."""
    name: str
    alias: list[str] = Field(default_factory=list)
    description: str = ""
    table: str
    field: str
    tags: list[str] = Field(default_factory=list)


class JoinRelation(BaseModel):
    """表间关联关系."""
    left_table: str
    left_field: str
    right_table: str
    right_field: str
    join_type: Literal["INNER", "LEFT", "RIGHT"] = "LEFT"


# ── 中间表示 (IR) ────────────────────────────────────────────

class IntermediateRepresentation(BaseModel):
    """语义层输出的中间表示——不直接生成 SQL，先生成 IR."""
    metrics: list[str] = Field(default_factory=list, description="指标名列表")
    dimensions: list[str] = Field(default_factory=list, description="维度名列表")
    filters: list[dict[str, Any]] = Field(default_factory=list)
    order_by: list[dict[str, Any]] = Field(default_factory=list)
    limit: int | None = None
    time_range: dict[str, str | None] | None = None  # {"start": "2024-01-01", "end": "2024-01-31"}

    def is_valid(self) -> tuple[bool, str]:
        if not self.metrics:
            return False, "No metrics specified in IR"
        return True, ""


# ── Plan 模式 ───────────────────────────────────────────────

class PlanStep(BaseModel):
    """计划中的单个步骤."""
    step_id: str = Field(default_factory=lambda: f"s_{uuid.uuid4().hex[:8]}")
    action: str  # retrieve_metrics, join_tables, apply_filter, sort, export
    description: str
    reason: str = ""  # 为什么需要这个步骤
    risk_level: Literal["low", "medium", "high"] = "low"
    estimated_tables: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class RiskFlag(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QueryPlan(BaseModel):
    """完整的查询计划——在执行前生成，高风险需人工确认."""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    user_query: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    overall_risk: RiskFlag = RiskFlag.LOW
    estimated_tables: list[str] = Field(default_factory=list)
    estimated_rows: int | None = None
    human_confirmation_required: bool = False
    confirmed: bool = False
    rejected_reason: str = ""


# ── Agent State ─────────────────────────────────────────────

class AgentState(BaseModel):
    """LangGraph Agent 的全局状态——跨节点传递."""
    # 输入
    user_query: str = ""
    user_id: str = "anonymous"

    # 计划阶段
    plan: QueryPlan | None = None
    plan_approved: bool | None = None  # None=待确认, True=已批准, False=已驳回

    # 语义层解析
    ir: IntermediateRepresentation | None = None
    sql: str | None = None
    sql_explain: str = ""

    # 校验结果
    syntax_valid: bool = False
    syntax_error: str = ""
    validation_results: list[dict[str, Any]] = Field(default_factory=list)
    validation_passed: bool = False

    # 执行
    query_result: list[dict[str, Any]] | None = None
    query_error: str = ""
    execution_time_ms: float = 0.0

    # 可观测
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    node_history: list[str] = Field(default_factory=list)  # 记录经过的节点
    quality_metrics: dict[str, Any] = Field(default_factory=dict)

    # 对话历史
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# ── Agent Output ────────────────────────────────────────────

class AgentOutput(BaseModel):
    """Agent 最终输出."""
    trace_id: str
    user_query: str
    plan: QueryPlan | None = None
    sql: str | None = None
    sql_explain: str = ""
    validation_summary: dict[str, Any] = Field(default_factory=dict)
    result: list[dict[str, Any]] | None = None
    result_summary: str = ""
    error: str | None = None
    execution_time_ms: float = 0.0
    node_path: list[str] = Field(default_factory=list)
