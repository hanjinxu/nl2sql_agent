# NL2SQL Data Agent 🤖

生产级 NL2SQL Agent —— **语义层 + Plan 模式 + 人在回路 (Human-in-the-Loop)**

## 架构

```
用户提问
   │
   ▼
┌──────────┐     ┌──────────────┐
│  Plan    │────→│ Human-in-    │  ← 高风险计划等待人工确认
│  LLM     │     │ the-Loop     │
└──────────┘     └──────┬───────┘
                        │ 已批准
                        ▼
                 ┌──────────────┐
                 │  Semantic    │  ← LLM 解析 → 中间表示 (IR)
                 │  Parser      │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  SQL Gen     │  ← 确定性 IR→SQL 翻译（不依赖 LLM）
                 │  (determin.) │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  4-Fold      │  ← 语法检查 + 安全扫描 + 结果检测
                 │  Validation  │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  Executor    │  ← SQL 执行
                 └──────────────┘
```

## 核心设计（来自千问 Data Agent 落地方案）

| 设计原则 | 实现 |
|---------|------|
| **语义层** — LLM 不直接写 SQL | LLM 生成受控的中间表示 (IR)，由确定性 SQL 生成器拼装 SQL |
| **Plan 模式** — 每次问数先出计划 | Planner Node 分析意图、生成步骤、风险分级 |
| **人在回路** — 高风险强制人工确认 | Human-confirm Node 拦截跨域 JOIN、全表扫描等操作 |
| **四重校验** | 语法静态检查 + 安全扫描 + 风险估计 + 结果异常检测 |
| **全链路可观测** | Tracing + 质量指标（一次命中率、人工介入率、响应时间） |

## 快速开始

### 1. 安装

```bash
cd nl2sql-data-agent
pip install -e ".[dev]"
```

### 2. 配置 API Key

```bash
export OPENAI_API_KEY="sk-xxx"
# 或使用通义千问
export OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 3. 运行

```bash
# ▶️ Web UI
nl2sql-agent --web

# 或直接使用 Python
python -m nl2sql_agent.main --web

# ▶️ 演示脚本
python -m nl2sql_agent.main --demo

# ▶️ 单次查询
nl2sql-agent --query "2024年1月第一周的每日GMV是多少？"
```

访问 http://localhost:8000 打开 Web 界面

### 4. API

```bash
# 自然语言查询
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "本月各城市的GMV排名"}'

# 查看质量指标
curl http://localhost:8000/metrics

# 查看语义层注册表
curl http://localhost:8000/schema
```

## 项目结构

```
src/nl2sql_agent/
├── models/          # 数据模型：AgentState, IR, QueryPlan
├── semantic/        # 语义层：指标/维度注册表 + IR 生成器
├── plan/            # Plan 模式：意图识别 + 计划生成 + 风险分级
├── sql_gen/         # SQL 生成：LLM→IR→SQL 确定性翻译
├── validation/      # 四重校验：语法 + 安全 + 风险 + 结果检测
├── observability/   # 可观测：Tracing + 质量指标采集
├── graph.py         # LangGraph 编排：StateGraph + Nodes + Edges
├── web.py           # FastAPI Web 服务 + UI
└── main.py          # CLI 入口
```

## 示例 — 电商场景

预置的语义层包含电商分析场景：

| 指标 | 业务名称 | SQL 映射 |
|------|---------|---------|
| `gmv` | GMV / 成交额 / 销售额 | SUM(orders.total_amount) |
| `order_count` | 订单量 / 订单数 | COUNT_DISTINCT(orders.order_id) |
| `avg_order_value` | 客单价 / AOV | AVG(orders.total_amount) |
| `item_count` | 商品销量 | SUM(order_items.quantity) |
| `refund_rate` | 退款率 | AVG(orders.is_refunded) |

| 维度 | 业务名称 | 来源表 |
|------|---------|-------|
| `date` | 日期 / 天 | orders.order_date |
| `city` | 城市 / 地区 | users.city |
| `product_category` | 商品类目 / 品类 | order_items.category |

## License

MIT
