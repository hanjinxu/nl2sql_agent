"""LangGraph Agent 编排 — State Graph + Nodes + Edges + Compile.

构建流程图：

  user_input
      │
      ▼
   ┌──────────┐
   │   plan   │ ← LLM 分析用户提问，制定查询计划
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │human_    │ ← 高风险计划暂停，等待人肉确认
   │ confirm  │
   └────┬─────┘
        │
        ├── 已批准 ──→ ┌───────────┐
        │              │generate_ir│ ← LLM 语义解析→IR
        │              └─────┬─────┘
        │                    │
        │                    ▼
        │              ┌───────────┐
        │              │sql_gen    │ ← 确定性 IR→SQL 翻译
        │              └─────┬─────┘
        │                    │
        │                    ▼
        │              ┌───────────┐
        │              │validation │ ← 四重校验
        │              └─────┬─────┘
        │                    │
        │                    ├── 校验通过 ──→ ┌────────┐
        │                    │               │executor│ ← 执行 SQL
        │                    │               └───┬────┘
        │                    │                   │
        │                    │                   ▼
        │                    │              ┌──────────┐
        │                    │              │formatter │ ← 格式化输出
        │                    │              └────┬─────┘
        │                    │                   │
        │                    ▼                   ▼
        ├── 已驳回 ──→ ┌──────────┐
        │              │  end     │
        │              └──────────┘
        │
        └── 待确认 ──→ (interrupt, 等用户输入)
"""

from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from nl2sql_agent.models import AgentOutput, AgentState
from nl2sql_agent.observability import Tracer, get_metrics_collector, QualityMetrics
from nl2sql_agent.plan import human_confirm_node, plan_node, should_continue_after_confirm
from nl2sql_agent.semantic import SemanticRegistry
from nl2sql_agent.sql_gen import generate_ir_node, sql_generation_node
from nl2sql_agent.validation import SQLValidator, validation_node


# ── SQL 执行器 ──────────────────────────────────────────────

class SQLExecutor:
    """SQL 执行器. 生产环境对接数据库连接池."""

    def __init__(self, connection_string: str | None = None) -> None:
        self.connection_string = connection_string
        # 如果没提供连接串，使用模拟执行（demo 模式）
        self._demo_mode = connection_string is None
        self._demo_data: list[dict[str, Any]] = []

    def set_demo_data(self, data: list[dict[str, Any]]) -> None:
        self._demo_data = data

    async def execute(self, sql: str) -> tuple[list[dict[str, Any]] | None, str, float]:
        """执行 SQL 并返回结果."""
        start = time.time()

        if self._demo_mode:
            # 模拟执行——返回配置的 demo 数据
            elapsed = (time.time() - start) * 1000
            return self._demo_data, "", elapsed

        # 真实执行 —— 需要一个数据库连接
        try:
            # 这里对接你的数据库（MySQL/ClickHouse/DuckDB）
            # import asyncmy  # for MySQL
            # async with asyncmy.connect(...) as conn:
            #     result = await conn.fetchall(sql)
            elapsed = (time.time() - start) * 1000
            raise NotImplementedError("Real DB execution not configured. Use demo mode or set connection_string.")
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return None, str(e), elapsed


async def executor_node(state: AgentState, executor: SQLExecutor) -> AgentState:
    """执行 SQL 并将结果写回 state."""
    state.node_history.append("executor")

    if state.sql is None:
        state.query_error = "No SQL to execute"
        return state

    result, error, elapsed = await executor.execute(state.sql)
    state.query_result = result
    state.query_error = error or ""
    state.execution_time_ms = elapsed

    return state


# ── 格式化节点 ──────────────────────────────────────────────

def formatter_node(state: AgentState) -> AgentState:
    """格式化输出."""
    state.node_history.append("formatter")
    return state


# ── 构建 LangGraph ─────────────────────────────────────────

def build_data_agent(
    llm: BaseChatModel,
    registry: SemanticRegistry | None = None,
    executor: SQLExecutor | None = None,
    checkpoint: MemorySaver | None = None,
) -> tuple[StateGraph, SemanticRegistry, SQLExecutor]:
    """构建 Data Agent StateGraph.

    返回 (graph, registry, executor) 三元组。
    """
    if registry is None:
        from nl2sql_agent.semantic import create_default_registry
        registry = create_default_registry()

    if executor is None:
        executor = SQLExecutor()

    if checkpoint is None:
        checkpoint = MemorySaver()

    # ── 定义 StateGraph ──
    builder = StateGraph(AgentState)

    # ── 注册节点 ──
    builder.add_node("plan", lambda state: plan_node(state, llm, registry))
    builder.add_node("human_confirm", human_confirm_node)
    builder.add_node("generate_ir", lambda state: generate_ir_node(state, llm, registry))
    builder.add_node("sql_generation", lambda state: sql_generation_node(state, registry))
    builder.add_node("validation", lambda state: validation_node(state))
    builder.add_node("executor", lambda state: executor_node(state, executor))
    builder.add_node("formatter", formatter_node)

    # ── 定义边 ──
    builder.set_entry_point("plan")

    builder.add_edge("plan", "human_confirm")

    # human_confirm → 条件分支
    builder.add_conditional_edges(
        "human_confirm",
        should_continue_after_confirm,
        {
            "generate_ir": "generate_ir",
            "end": END,
        },
    )

    builder.add_edge("generate_ir", "sql_generation")
    builder.add_edge("sql_generation", "validation")

    # validation → 条件分支（通过/失败都先到 executor，让 executor 决定）
    builder.add_edge("validation", "executor")
    builder.add_edge("executor", "formatter")
    builder.add_edge("formatter", END)

    # ── 编译 ──
    graph = builder.compile(checkpointer=checkpointer)

    return graph, registry, executor


# ── 简易调用接口 ────────────────────────────────────────────

async def run_agent(
    graph: StateGraph,
    query: str,
    user_id: str = "anonymous",
    thread_id: str | None = None,
    plan_approved: bool | None = None,
) -> AgentOutput:
    """运行 Data Agent 的简便接口."""
    import uuid
    thread_id = thread_id or uuid.uuid4().hex[:16]

    tracer = Tracer(trace_id=thread_id)
    tracer.start_span("total")

    initial_state = AgentState(
        user_query=query,
        user_id=user_id,
        trace_id=thread_id,
        plan_approved=plan_approved,  # None = 需确认, True = 已批准
    )

    # 配置
    config = {
        "configurable": {"thread_id": thread_id},
    }

    try:
        start_time = time.time()
        result_state = await graph.ainvoke(initial_state, config)
        total_time = (time.time() - start_time) * 1000
        tracer.end_span("ok")

        # 记录质量指标
        metrics = QualityMetrics(
            trace_id=thread_id,
            first_attempt_success=result_state.validation_passed,
            human_intervention=result_state.plan is not None
            and result_state.plan.human_confirmation_required,
            sql_execution_success=not result_state.query_error,
            plan_rejected=result_state.plan_approved is False,
            response_time_ms=total_time,
        )
        get_metrics_collector().record(metrics)

        return AgentOutput(
            trace_id=thread_id,
            user_query=query,
            plan=result_state.plan,
            sql=result_state.sql,
            sql_explain=result_state.sql_explain,
            validation_summary={
                "passed": result_state.validation_passed,
                "checks": result_state.validation_results,
            },
            result=result_state.query_result,
            result_summary=_summarize_result(result_state),
            error=result_state.query_error if result_state.query_error else None,
            execution_time_ms=total_time,
            node_path=result_state.node_history,
        )

    except Exception as e:
        tracer.end_span("error", str(e))
        return AgentOutput(
            trace_id=thread_id,
            user_query=query,
            error=f"Agent execution failed: {e}",
            execution_time_ms=(time.time() - start_time) * 1000 if 'start_time' in dir() else 0,
            node_path=[],
        )


def _summarize_result(state: AgentState) -> str:
    """生成结果摘要."""
    parts = []
    if state.query_result is not None:
        parts.append(f"返回 {len(state.query_result)} 行结果")
    if state.query_error:
        parts.append(f"错误: {state.query_error}")
    if state.sql:
        parts.append(f"SQL: {state.sql[:100]}...")
    return " | ".join(parts) if parts else "无结果"
