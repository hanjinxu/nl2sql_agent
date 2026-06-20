"""SQL 生成模块 — IR → SQL 翻译器 + 语义层拼装.

核心设计理念：
  LLM 生成受控的中间表示 (IR)，然后由 SQL Generator 拼装 SQL。
  这里的拼装逻辑是确定性的，不依赖 LLM 写 SQL，
  这是压低幻觉的关键。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from nl2sql_agent.models import AgentState, IntermediateRepresentation
from nl2sql_agent.semantic import SemanticRegistry


# ── 确定性 IR → SQL 翻译器 ──────────────────────────────────

class SQLGenerator:
    """基于 IR 的确定性 SQL 生成器."""

    def __init__(self, registry: SemanticRegistry) -> None:
        self.registry = registry

    def generate(self, ir: IntermediateRepresentation) -> str:
        """从 IR 生成 SQL."""
        metrics = self._resolve_metrics(ir.metrics)
        dimensions = self._resolve_dimensions(ir.dimensions)
        tables = self._resolve_tables(ir.metrics, ir.dimensions)
        joins = self._build_joins(tables)
        where = self._build_where(ir.filters, ir.time_range)
        group_by = self._build_group_by(ir.dimensions)
        order_by = self._build_order_by(ir.order_by)
        limit = ir.limit

        select_items = metrics + dimensions

        sep = ",\\n  "
        sql_parts = [f"SELECT\n  {sep.join(select_items)}"]
        sql_parts.append(f"FROM {tables[0]}")

        if joins:
            sql_parts.append("\n".join(joins))

        if where:
            sql_parts.append(f"WHERE {' AND '.join(where)}")

        if group_by:
            sql_parts.append(f"GROUP BY {', '.join(group_by)}")

        if order_by:
            sql_parts.append(f"ORDER BY {', '.join(order_by)}")

        if limit is not None:
            sql_parts.append(f"LIMIT {limit}")

        return "\n".join(sql_parts)

    def _resolve_metrics(self, metric_names: list[str]) -> list[str]:
        """将指标名解析为 SQL 表达式."""
        result = []
        for name in metric_names:
            m = self.registry.metrics.get(name)
            if m is None:
                result.append(f"-- Unknown metric: {name}")
                continue
            alias = name
            expr = f"{m.aggregation.value}({m.table}.{m.field}) AS {alias}"
            result.append(expr)
        return result

    def _resolve_dimensions(self, dim_names: list[str]) -> list[str]:
        result = []
        for name in dim_names:
            d = self.registry.dimensions.get(name)
            if d is None:
                result.append(f"-- Unknown dimension: {name}")
                continue
            result.append(f"{d.table}.{d.field} AS {name}")
        return result

    def _resolve_tables(self, metric_names: list[str], dim_names: list[str]) -> list[str]:
        """确定需要查询哪些表，按 join 顺序排列."""
        tables: set[str] = set()

        for name in metric_names:
            m = self.registry.metrics.get(name)
            if m and m.table:
                tables.add(m.table)

        for name in dim_names:
            d = self.registry.dimensions.get(name)
            if d and d.table:
                tables.add(d.table)

        # 尝试排序——第一个表作为主表
        sorted_tables = list(tables)
        if "orders" in sorted_tables:
            sorted_tables.remove("orders")
            sorted_tables.insert(0, "orders")
        elif sorted_tables:
            pass  # 保持原样

        return sorted_tables if sorted_tables else ["orders"]

    def _build_joins(self, tables: list[str]) -> list[str]:
        """按注册的关联关系生成 JOIN 子句."""
        if len(tables) <= 1:
            return []

        joins = []
        for i in range(1, len(tables)):
            left = tables[i - 1]
            right = tables[i]

            # 查注册的 join 关系
            for jr in self.registry.join_relations:
                if jr.left_table == left and jr.right_table == right:
                    joins.append(
                        f"{jr.join_type} JOIN {jr.right_table}\n"
                        f"  ON {jr.left_table}.{jr.left_field} = {jr.right_table}.{jr.right_field}"
                    )
                    break
                elif jr.right_table == left and jr.left_table == right:
                    joins.append(
                        f"{jr.join_type} JOIN {jr.right_table}\n"
                        f"  ON {jr.left_table}.{jr.left_field} = {jr.right_table}.{jr.right_field}"
                    )
                    break
            else:
                # 用表上定义的 join_keys 兜底
                right_table_obj = self.registry.tables.get(right)
                if right_table_obj and left in right_table_obj.join_keys:
                    key = right_table_obj.join_keys[left]
                    joins.append(
                        f"LEFT JOIN {right}\n  ON {left}.{key} = {right}.{key}"
                    )

        return joins

    def _build_where(
        self,
        filters: list[dict[str, Any]],
        time_range: dict[str, str | None] | None = None,
    ) -> list[str]:
        clauses = []

        for f in filters:
            field = f.get("field", "")
            op = f.get("operator", "=")
            value = f.get("value", "")

            if isinstance(value, str):
                value = f"'{value}'"

            clauses.append(f"{field} {op} {value}")

        if time_range:
            if time_range.get("start"):
                clauses.append(f"orders.order_date >= '{time_range['start']}'")
            if time_range.get("end"):
                clauses.append(f"orders.order_date <= '{time_range['end']}'")

        return clauses

    def _build_group_by(self, dimensions: list[str]) -> list[str]:
        return [d for d in dimensions if d in self.registry.dimensions]

    def _build_order_by(self, order_by: list[dict[str, Any]]) -> list[str]:
        items = []
        for o in order_by:
            field = o.get("field", "")
            direction = o.get("direction", "DESC").upper()
            if direction not in ("ASC", "DESC"):
                direction = "DESC"
            items.append(f"{field} {direction}")
        return items


# ── LLM 辅助 IR 生成 ────────────────────────────────────────

IR_GENERATION_PROMPT = """你是 Data Agent 的语义解析器（Semantic Parser）。

你的任务是将用户的自然语言提问，映射到语义层的**中间表示（IR）**——使用已注册的指标和维度名称。

## 输出格式
返回 JSON，包含：
```json
{
  "metrics": ["指标名1", "指标名2"],
  "dimensions": ["维度名1"],
  "filters": [{"field": "表名.字段", "operator": "=", "value": "xxx"}],
  "order_by": [{"field": "metrics_or_dim", "direction": "DESC"}],
  "limit": 10,
  "time_range": {"start": "2024-01-01", "end": null}
}
```

## 规则
1. 只能使用语义层中已注册的**指标名**和**维度名**
2. 如果用户提到了语义层中不存在的名称，选择最接近的已注册名称
3. 除非用户明确要求具体数量，否则不设置 limit
4. 时间范围从用户提问中推断（"本月"→当前月，"上周"→上周）
5. 默认排序按指标降序"""


async def llm_ir_generation(
    query: str,
    plan: dict[str, Any] | None,
    schema_context: str,
    llm: BaseChatModel,
) -> IntermediateRepresentation:
    """用 LLM 将用户提问转为中间表示."""
    plan_context = ""
    if plan:
        plan_context = f"\n## 已制定的查询计划\n{json.dumps(plan, ensure_ascii=False, indent=2)}"

    messages = [
        SystemMessage(content=IR_GENERATION_PROMPT),
        HumanMessage(content=f"## 用户提问\n{query}\n{plan_context}\n\n## 可用数据\n{schema_context}\n\n输出 JSON 格式的中间表示："),
    ]

    response = await llm.ainvoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    # 解析 JSON
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = content.strip()
    json_str = re.sub(r"^[^{]*", "", json_str)
    json_str = re.sub(r"[^}]*$", "", json_str)

    data: dict[str, Any] = json.loads(json_str)

    return IntermediateRepresentation(
        metrics=data.get("metrics", []),
        dimensions=data.get("dimensions", []),
        filters=data.get("filters", []),
        order_by=data.get("order_by", []),
        limit=data.get("limit"),
        time_range=data.get("time_range"),
    )


# ── LangGraph Nodes ─────────────────────────────────────────

async def generate_ir_node(state: AgentState, llm: BaseChatModel, registry: SemanticRegistry) -> AgentState:
    """用 LLM 把用户提问解析成 IR."""
    schema_prompt = registry.get_table_schema_prompt()
    plan_dict = state.plan.model_dump() if state.plan else None

    ir = await llm_ir_generation(state.user_query, plan_dict, schema_prompt, llm)
    state.ir = ir
    state.node_history.append("generate_ir")

    # 校验 IR
    valid, msg = ir.is_valid()
    if not valid:
        state.query_error = f"IR validation failed: {msg}"

    return state


async def sql_generation_node(state: AgentState, registry: SemanticRegistry) -> AgentState:
    """将 IR 拼装为 SQL."""
    if state.ir is None:
        state.query_error = "No IR to generate SQL from"
        state.node_history.append("sql_generation")
        return state

    generator = SQLGenerator(registry)
    sql = generator.generate(state.ir)
    state.sql = sql
    state.node_history.append("sql_generation")
    return state
