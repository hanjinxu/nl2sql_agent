"""Semantic Layer — 指标/维度注册表 + IR 生成器.

核心设计理念：
  LLM 不直接写 SQL，而是通过语义层进行指标/维度召回，
  生成受控的中间表示 (IR)，再由 IR 拼装 SQL。
  这是压低幻觉、提高一致性的关键路径。
"""

from __future__ import annotations

from typing import Any

from nl2sql_agent.models import (
    AggFunction,
    DimensionDefinition,
    IntermediateRepresentation,
    JoinRelation,
    MetricDefinition,
    SchemaTable,
)


class SemanticRegistry:
    """语义注册表——管理所有指标、维度、表和关联关系.

    相当于"指标语义仓" + "数据集目录" 的 in-memory 实现.
    生产环境中应替换为数据库/配置中心.
    """

    def __init__(self) -> None:
        self.metrics: dict[str, MetricDefinition] = {}
        self.dimensions: dict[str, DimensionDefinition] = {}
        self.tables: dict[str, SchemaTable] = {}
        self.join_relations: list[JoinRelation] = []

    # ── 注册方法 ──────────────────────────────────────────

    def register_metric(
        self,
        name: str,
        alias: list[str] | None = None,
        description: str = "",
        table: str = "",
        field: str = "",
        aggregation: str | AggFunction = AggFunction.SUM,
        filters: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
    ) -> MetricDefinition:
        if isinstance(aggregation, str):
            aggregation = AggFunction(aggregation.upper())
        m = MetricDefinition(
            name=name,
            alias=alias or [],
            description=description,
            table=table,
            field=field,
            aggregation=aggregation,
            filters=filters or [],
            tags=tags or [],
        )
        self.metrics[name] = m
        return m

    def register_dimension(
        self,
        name: str,
        alias: list[str] | None = None,
        description: str = "",
        table: str = "",
        field: str = "",
        tags: list[str] | None = None,
    ) -> DimensionDefinition:
        d = DimensionDefinition(
            name=name,
            alias=alias or [],
            description=description,
            table=table,
            field=field,
            tags=tags or [],
        )
        self.dimensions[name] = d
        return d

    def register_table(self, table: SchemaTable) -> SchemaTable:
        self.tables[table.name] = table
        return table

    def add_join(
        self,
        left_table: str,
        left_field: str,
        right_table: str,
        right_field: str,
        join_type: str = "LEFT",
    ) -> JoinRelation:
        j = JoinRelation(
            left_table=left_table,
            left_field=left_field,
            right_table=right_table,
            right_field=right_field,
            join_type=join_type,  # type: ignore
        )
        self.join_relations.append(j)
        return j

    # ── 检索方法 ──────────────────────────────────────────

    def search_metrics(self, query: str) -> list[MetricDefinition]:
        """模糊搜索指标——匹配 name 和 alias."""
        q = query.lower()
        results = []
        for m in self.metrics.values():
            if q in m.name.lower():
                results.append(m)
                continue
            for a in m.alias:
                if q in a.lower():
                    results.append(m)
                    break
        return results

    def search_dimensions(self, query: str) -> list[DimensionDefinition]:
        q = query.lower()
        results = []
        for d in self.dimensions.values():
            if q in d.name.lower():
                results.append(d)
                continue
            for a in d.alias:
                if q in a.lower():
                    results.append(d)
                    break
        return results

    def find_join_path(self, tables: list[str]) -> list[JoinRelation]:
        """找出连接一组表所需的 JOIN 路径."""
        needed: list[JoinRelation] = []
        remaining = set(tables)

        # 简单实现：逐个表找连接
        for jr in self.join_relations:
            if jr.left_table in remaining and jr.right_table in remaining:
                needed.append(jr)
                remaining.discard(jr.left_table)
                remaining.discard(jr.right_table)

        return needed

    def get_table_schema_prompt(self) -> str:
        """生成给 LLM 的表结构提示——让模型理解可用的数据."""
        lines = ["## 可用数据集", ""]
        for t_name, table in self.tables.items():
            lines.append(f"### {t_name}")
            if table.description:
                lines.append(f"描述: {table.description}")
            lines.append("字段:")
            for f_name, field in table.fields.items():
                alias_str = f" (别名: {', '.join(field.alias)})" if field.alias else ""
                lines.append(f"  - {f_name}: {field.type.value}{alias_str}")
            lines.append("")

        lines.append("## 可用指标")
        for m_name, m in self.metrics.items():
            alias_str = f" (别名: {', '.join(m.alias)})" if m.alias else ""
            lines.append(f"  - {m_name}{alias_str}: {m.description} → {m.aggregation.value}({m.table}.{m.field})")

        lines.append("")
        lines.append("## 可用维度")
        for d_name, d in self.dimensions.items():
            alias_str = f" (别名: {', '.join(d.alias)})" if d.alias else ""
            lines.append(f"  - {d_name}{alias_str}: {d.description} → {d.table}.{d.field}")

        return "\n".join(lines)

    # ── IR 生成 ────────────────────────────────────────────

    def build_ir(
        self,
        metrics: list[str],
        dimensions: list[str] | None = None,
        filters: list[dict[str, Any]] | None = None,
        order_by: list[dict[str, Any]] | None = None,
        limit: int | None = None,
        time_range: dict[str, str | None] | None = None,
    ) -> IntermediateRepresentation:
        """校验并构建中间表示."""
        # 校验指标存在
        for m in metrics:
            if m not in self.metrics:
                raise ValueError(f"Unknown metric: {m}")

        # 校验维度存在
        for d in (dimensions or []):
            if d not in self.dimensions:
                raise ValueError(f"Unknown dimension: {d}")

        return IntermediateRepresentation(
            metrics=metrics,
            dimensions=dimensions or [],
            filters=filters or [],
            order_by=order_by or [],
            limit=limit,
            time_range=time_range,
        )


# ── 默认注册表（电商场景示例） ─────────────────────────────

def create_default_registry() -> SemanticRegistry:
    """创建一个电商分析场景的默认语义注册表."""
    reg = SemanticRegistry()

    # 注册表
    orders_table = SchemaTable(
        name="orders",
        alias=["订单表", "order"],
        description="订单主表，每一行是一笔订单",
        join_keys={"order_items": "order_id"},
    )
    reg.register_table(orders_table)

    items_table = SchemaTable(
        name="order_items",
        alias=["订单明细", "order_item"],
        description="订单商品明细，每一行是一个商品的购买记录",
        join_keys={"orders": "order_id", "products": "product_id"},
    )
    reg.register_table(items_table)

    users_table = SchemaTable(
        name="users",
        alias=["用户表", "user"],
        description="用户信息表",
        join_keys={"orders": "user_id"},
    )
    reg.register_table(users_table)

    # 注册指标
    reg.register_metric("gmv", ["GMV", "成交额", "销售额", "总金额"],
                        "订单总金额（实付金额）", "orders", "total_amount", "SUM",
                        tags=["核心指标", "营收"])
    reg.register_metric("order_count", ["订单量", "订单数", "总订单"],
                        "订单总数", "orders", "order_id", "COUNT_DISTINCT",
                        tags=["核心指标", "流量"])
    reg.register_metric("avg_order_value", ["客单价", "平均订单金额", "AOV"],
                        "平均每单金额", "orders", "total_amount", "AVG",
                        tags=["核心指标", "营收"])
    reg.register_metric("item_count", ["商品销量", "售出件数"],
                        "售出商品总件数", "order_items", "quantity", "SUM",
                        tags=["商品"])
    reg.register_metric("refund_rate", ["退款率"],
                        "退款订单占比", "orders", "is_refunded", "AVG",
                        tags=["风控"])
    reg.register_metric("new_user_count", ["新用户数", "新增用户"],
                        "新注册用户数", "users", "user_id", "COUNT_DISTINCT",
                        filters=[{"field": "is_new", "operator": "=", "value": True}],
                        tags=["用户"])

    # 注册维度
    reg.register_dimension("date", ["日期", "天", "day"],
                           "订单日期", "orders", "order_date",
                           tags=["时间"])
    reg.register_dimension("month", ["月份", "月"],
                           "订单月份", "orders", "order_month",
                           tags=["时间"])
    reg.register_dimension("city", ["城市", "地区"],
                           "用户所在城市", "users", "city",
                           tags=["地域"])
    reg.register_dimension("product_category", ["商品类目", "品类", "类目"],
                           "商品所属类目", "order_items", "category",
                           tags=["商品"])
    reg.register_dimension("payment_method", ["支付方式", "支付渠道"],
                           "支付方式", "orders", "payment_method",
                           tags=["交易"])

    # 注册表关联
    reg.add_join("orders", "order_id", "order_items", "order_id")
    reg.add_join("orders", "user_id", "users", "user_id")

    return reg
