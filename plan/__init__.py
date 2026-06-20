"""Plan 模式 — 意图识别 + 查询计划生成 + 风险分级.

核心设计理念：
  任何一次问数都必须有可解释的计划。
  对高风险动作（跨域 join、导出明细、全表扫描）必须强制人工确认。
  这是 Data Agent 生产化的第一道安全门。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from nl2sql_agent.models import (
    AgentState,
    PlanStep,
    QueryPlan,
    RiskFlag,
)
from nl2sql_agent.semantic import SemanticRegistry


# ── Prompt ─────────────────────────────────────────────────

PLAN_SYSTEM_PROMPT = """你是 Data Agent 的计划制定者（Planner）。

你的任务是根据用户的自然语言提问，制定一个可执行的查询计划。
计划必须遵循以下规则：

## 规则
1. **必须使用语义层定义的指标和维度名称**，不能自己编造
2. 每个步骤必须有明确的 action、description 和 risk_level
3. 风险分级标准：
   - low: 单表查询，仅聚合，无跨表 join
   - medium: 2-3 表 join，有过滤条件
   - high: 跨域 join（>3 表）、无过滤条件的全表扫描、导出明细数据
4. 高风险计划必须标记 human_confirmation_required=true
5. 如果用户意图不明确（提问模糊、歧义、缺少必要条件），在 plan 的 rejected_reason 中标注

## 输出格式
你返回 JSON 格式的计划，包含：
- user_query: 原始用户提问
- steps: 步骤列表 [{action, description, reason, risk_level, estimated_tables}]
- overall_risk: 整体风险 (low/medium/high)
- estimated_tables: 涉及的所有表
- human_confirmation_required: true/false

如果用户问题不明确或无法解析，steps 为空数组，在 rejected_reason 中说明原因。"""


def build_plan_prompt(query: str, schema_context: str) -> list:
    """构建 Plan 生成的 prompt."""
    return [
        SystemMessage(content=PLAN_SYSTEM_PROMPT),
        HumanMessage(content=f"## 用户提问\n{query}\n\n## 可用数据\n{schema_context}\n\n请输出 JSON 格式的查询计划。"),
    ]


def parse_plan_response(response: str) -> QueryPlan:
    """从 LLM 响应中解析 QueryPlan."""
    # 提取 JSON（LLM 可能用 ```json ``` 包裹）
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response.strip()

    # 清理可能的非 JSON 前缀/后缀
    json_str = re.sub(r"^[^{]*", "", json_str)
    json_str = re.sub(r"[^}]*$", "", json_str)

    data: dict[str, Any] = json.loads(json_str)

    steps = []
    for s in data.get("steps", []):
        steps.append(PlanStep(
            action=s.get("action", "unknown"),
            description=s.get("description", ""),
            reason=s.get("reason", ""),
            risk_level=s.get("risk_level", "low"),
            estimated_tables=s.get("estimated_tables", []),
        ))

    # 计算整体风险
    step_risks = [s.risk_level for s in steps]
    if "high" in step_risks:
        overall_risk = RiskFlag.HIGH
    elif "medium" in step_risks:
        overall_risk = RiskFlag.MEDIUM
    else:
        overall_risk = RiskFlag.LOW

    return QueryPlan(
        user_query=data.get("user_query", ""),
        steps=steps,
        overall_risk=overall_risk,
        estimated_tables=data.get("estimated_tables", []),
        estimated_rows=data.get("estimated_rows"),
        human_confirmation_required=overall_risk in (RiskFlag.HIGH, RiskFlag.MEDIUM)
        or data.get("human_confirmation_required", False),
        rejected_reason=data.get("rejected_reason", ""),
    )


# ── LangGraph Node ─────────────────────────────────────────

async def plan_node(state: AgentState, llm: BaseChatModel, registry: SemanticRegistry) -> AgentState:
    """Plan Node — 分析用户提问，生成查询计划."""
    schema_prompt = registry.get_table_schema_prompt()
    messages = build_plan_prompt(state.user_query, schema_prompt)

    response = await llm.ainvoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    try:
        plan = parse_plan_response(content)
    except (json.JSONDecodeError, KeyError) as e:
        # 解析失败，回退为简单计划
        plan = QueryPlan(
            user_query=state.user_query,
            steps=[PlanStep(
                action="direct_query",
                description=f"直接执行查询: {state.user_query}",
                reason="Plan parsing failed, falling back to direct query",
                risk_level="medium",
            )],
            overall_risk=RiskFlag.MEDIUM,
            human_confirmation_required=True,
            rejected_reason=f"Plan parsing error: {e}. Falling back to direct query.",
        )

    state.plan = plan
    state.node_history.append("plan")

    # 如果计划不可用（被驳回），标记
    if plan.rejected_reason and not plan.steps:
        state.plan_approved = False

    return state


# ── 人工确认模拟 ─────────────────────────────────────────────

def human_confirm_node(state: AgentState) -> AgentState:
    """人工确认 Node — 高风险计划需用户确认后才能继续.

    在交互式 CLI/Web 环境中，此节点会暂停等待用户输入。
    这里我们提供一个简单的模拟接口。
    """
    state.node_history.append("human_confirm")

    if state.plan is None:
        state.plan_approved = False
        return state

    if not state.plan.human_confirmation_required:
        # 低风险，自动通过
        state.plan_approved = True
        return state

    # 如果已经确认过（从 Web 接口传入），直接使用
    if state.plan_approved is not None:
        return state

    # 默认等待确认——在实际应用中这里会 breakpoint
    # LangGraph 的 interrupt() 机制会在 Web 环境中使用
    state.plan_approved = None  # 待确认
    return state


def should_continue_after_confirm(state: AgentState) -> str:
    """条件边决策：计划被批准则继续，否则结束."""
    if state.plan_approved is True:
        return "generate_ir"
    # plan_approved 为 None 时，等待人工确认（interrupt 点）
    return "end"
