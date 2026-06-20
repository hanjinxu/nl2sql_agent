"""FastAPI Web Server — Data Agent 交互接口.

提供 Web UI 和 REST API，支持：
- POST /query: 提交自然语言查询
- GET /plan/{plan_id}: 查看查询计划
- POST /plan/{plan_id}/confirm: 人工确认/驳回计划
- GET /metrics: 查看质量指标
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from nl2sql_agent.graph import build_data_agent, run_agent
from nl2sql_agent.models import AgentOutput
from nl2sql_agent.observability import get_metrics_collector
from nl2sql_agent.semantic import SemanticRegistry, create_default_registry
from nl2sql_agent.sql_gen import SQLExecutor


# ── 请求/响应模型 ──────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    plan_approved: bool | None = None  # None=等待确认, True=已批准


class ConfirmRequest(BaseModel):
    plan_id: str
    approved: bool
    reason: str = ""


class AgentResponse(BaseModel):
    success: bool
    data: AgentOutput | None = None
    message: str = ""


# ── 全局状态 ────────────────────────────────────────────────

class AppState:
    def __init__(self) -> None:
        self.registry: SemanticRegistry = create_default_registry()
        self.executor = SQLExecutor()
        self.llm: ChatOpenAI | None = None
        self.graph = None
        self.pending_plans: dict[str, AgentOutput] = {}

    def setup_demo_data(self) -> None:
        self.executor.set_demo_data([
            {"date": "2024-01-01", "gmv": 1250000.0, "order_count": 850},
            {"date": "2024-01-02", "gmv": 1180000.0, "order_count": 790},
            {"date": "2024-01-03", "gmv": 1320000.0, "order_count": 920},
            {"date": "2024-01-04", "gmv": 1410000.0, "order_count": 980},
            {"date": "2024-01-05", "gmv": 1520000.0, "order_count": 1050},
            {"date": "2024-01-06", "gmv": 980000.0, "order_count": 720},
            {"date": "2024-01-07", "gmv": 890000.0, "order_count": 650},
        ])


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭生命周期."""
    # 启动
    try:
        app_state.llm = ChatOpenAI(
            model="qwen-plus",
            temperature=0.1,
        )
    except Exception as e:
        print(f"⚠️  LLM init warning: {e}")
        app_state.llm = None

    app_state.setup_demo_data()

    if app_state.llm:
        graph, _, _ = build_data_agent(app_state.llm, app_state.registry, app_state.executor)
        app_state.graph = graph

    yield
    # 关闭


app = FastAPI(
    title="NL2SQL Data Agent",
    description="Production-grade Data Agent with Semantic Layer + Plan Mode + Human-in-the-Loop",
    version="0.1.0",
    lifespan=lifespan,
)


# ── API 路由 ────────────────────────────────────────────────

@app.post("/query", response_model=AgentResponse)
async def query(request: QueryRequest):
    """提交自然语言查询."""
    if app_state.graph is None:
        return AgentResponse(success=False, message="Agent not initialized (check LLM config)")

    try:
        output = await run_agent(
            app_state.graph,
            request.query,
            user_id=request.user_id,
            plan_approved=request.plan_approved,
        )

        # 如果计划待确认，暂存
        if output.plan and output.plan.human_confirmation_required and not output.plan.confirmed:
            app_state.pending_plans[output.plan.plan_id] = output

        return AgentResponse(
            success=output.error is None,
            data=output,
            message=output.error or "OK",
        )
    except Exception as e:
        return AgentResponse(success=False, message=str(e))


@app.get("/plan/{plan_id}")
async def get_plan(plan_id: str):
    """查看指定计划的详情."""
    output = app_state.pending_plans.get(plan_id)
    if output is None:
        raise HTTPException(status_code=404, detail="Plan not found")
    return AgentResponse(success=True, data=output)


@app.post("/plan/{plan_id}/confirm")
async def confirm_plan(plan_id: str, request: ConfirmRequest):
    """人工确认或驳回计划."""
    output = app_state.pending_plans.get(plan_id)
    if output is None:
        raise HTTPException(status_code=404, detail="Plan not found")

    # 更新 plan 状态
    if output.plan:
        output.plan.confirmed = request.approved
        output.plan.rejected_reason = request.reason if not request.approved else ""

    # 重新执行（带确认结果）
    if app_state.graph and output.plan:
        new_output = await run_agent(
            app_state.graph,
            output.user_query,
            user_id="anonymous",
            plan_approved=request.approved,
        )
        app_state.pending_plans.pop(plan_id, None)
        return AgentResponse(
            success=new_output.error is None,
            data=new_output,
            message="Plan confirmed, query executed" if request.approved else "Plan rejected",
        )

    return AgentResponse(success=False, message="Agent not initialized")


@app.get("/metrics")
async def get_metrics():
    """查看质量指标."""
    collector = get_metrics_collector()
    return {
        "summary": collector.get_summary(),
        "recent": [m.to_dict() for m in collector.metrics[-20:]],  # 最近20条
    }


@app.get("/schema")
async def get_schema():
    """查看语义层注册表."""
    return {
        "metrics": {k: v.model_dump() for k, v in app_state.registry.metrics.items()},
        "dimensions": {k: v.model_dump() for k, v in app_state.registry.dimensions.items()},
        "tables": {k: v.model_dump() for k, v in app_state.registry.tables.items()},
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """简易 Web 界面."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>NL2SQL Data Agent</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f0f4f8; color: #1a202c; padding: 20px;
            }
            .container { max-width: 900px; margin: 0 auto; }
            h1 { color: #2d3748; margin-bottom: 8px; }
            .subtitle { color: #718096; font-size: 14px; margin-bottom: 24px; }
            .card {
                background: white; border-radius: 12px; padding: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px;
            }
            textarea {
                width: 100%; padding: 12px; border: 2px solid #e2e8f0;
                border-radius: 8px; font-size: 14px; resize: vertical;
                min-height: 80px; font-family: inherit;
            }
            textarea:focus { outline: none; border-color: #4299e1; }
            button {
                background: #4299e1; color: white; border: none;
                padding: 10px 24px; border-radius: 8px; font-size: 14px;
                cursor: pointer; transition: background 0.2s;
            }
            button:hover { background: #3182ce; }
            button:disabled { background: #a0aec0; cursor: not-allowed; }
            .result { margin-top: 16px; }
            .result pre {
                background: #1a202c; color: #e2e8f0; padding: 16px;
                border-radius: 8px; overflow-x: auto; font-size: 13px;
            }
            .error { color: #e53e3e; margin-top: 8px; }
            .badge {
                display: inline-block; padding: 2px 8px; border-radius: 4px;
                font-size: 12px; font-weight: 600;
            }
            .badge-low { background: #c6f6d5; color: #276749; }
            .badge-medium { background: #fefcbf; color: #975a16; }
            .badge-high { background: #fed7d7; color: #9b2c2c; }
            .plan-step {
                padding: 8px 12px; margin: 4px 0; background: #f7fafc;
                border-radius: 6px; border-left: 3px solid #4299e1;
            }
            .metrics-grid {
                display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px; margin-top: 12px;
            }
            .metric-card {
                background: #f7fafc; padding: 16px; border-radius: 8px; text-align: center;
            }
            .metric-card .value { font-size: 24px; font-weight: 700; color: #2d3748; }
            .metric-card .label { font-size: 12px; color: #718096; margin-top: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 NL2SQL Data Agent</h1>
            <p class="subtitle">语义层 + Plan 模式 + 人在回路 · 生产级 NL2SQL</p>

            <div class="card">
                <textarea id="query" placeholder="输入自然语言查询，例如：2024年1月第一周的每日GMV是多少？"></textarea>
                <div style="margin-top: 12px; display: flex; gap: 8px;">
                    <button onclick="submitQuery()">🚀 查询</button>
                    <button onclick="clearResult()" style="background: #718096;">清除</button>
                </div>
                <div id="loading" style="display:none; margin-top: 12px; color: #718096;">⏳ 处理中...</div>
            </div>

            <div class="card" id="resultCard" style="display:none;">
                <h3 style="margin-bottom: 12px;">📊 查询结果</h3>
                <div id="resultContent"></div>
            </div>

            <div class="card">
                <h3>📈 质量指标</h3>
                <button onclick="loadMetrics()" style="margin-top: 8px;">刷新指标</button>
                <div id="metricsContent"></div>
            </div>
        </div>

        <script>
            async function submitQuery() {
                const q = document.getElementById('query').value.trim();
                if (!q) return alert('请输入查询');
                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultCard').style.display = 'none';

                try {
                    const res = await fetch('/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: q}),
                    });
                    const data = await res.json();
                    renderResult(data);
                } catch (e) {
                    document.getElementById('resultContent').innerHTML =
                        '<div class="error">请求失败: ' + e.message + '</div>';
                }
                document.getElementById('loading').style.display = 'none';
                document.getElementById('resultCard').style.display = 'block';
            }

            function renderResult(data) {
                const content = document.getElementById('resultContent');
                if (!data.success) {
                    content.innerHTML = '<div class="error">❌ ' + (data.message || '未知错误') + '</div>';
                    return;
                }
                const d = data.data;
                let html = '';

                // 路径
                if (d.node_path) {
                    html += '<p style="color:#718096;font-size:13px;">📍 ' + d.node_path.join(' → ') + ' | ⏱ ' + d.execution_time_ms.toFixed(0) + 'ms</p>';
                }

                // 计划
                if (d.plan) {
                    html += '<h4 style="margin-top:12px;">📋 查询计划</h4>';
                    html += '<p>整体风险: <span class="badge badge-' + d.plan.overall_risk + '">' + d.plan.overall_risk + '</span>';
                    html += ' | 需确认: ' + d.plan.human_confirmation_required + '</p>';
                    for (const step of (d.plan.steps || [])) {
                        html += '<div class="plan-step">';
                        html += '<span class="badge badge-' + step.risk_level + '">' + step.risk_level + '</span> ';
                        html += '<strong>' + step.action + '</strong>: ' + step.description.slice(0, 80);
                        html += '</div>';
                    }
                }

                // SQL
                if (d.sql) {
                    html += '<h4 style="margin-top:12px;">🔍 SQL</h4>';
                    html += '<pre>' + escapeHtml(d.sql) + '</pre>';
                }

                // 校验
                if (d.validation_summary) {
                    const v = d.validation_summary;
                    html += '<p>✅ 校验: ' + (v.passed ? '通过' : '未通过') + '</p>';
                }

                // 结果
                if (d.result) {
                    html += '<h4 style="margin-top:12px;">📊 结果 (' + d.result.length + ' 行)</h4>';
                    html += '<pre>' + escapeHtml(JSON.stringify(d.result.slice(0, 10), null, 2)) + '</pre>';
                    if (d.result.length > 10) html += '<p>... 还有 ' + (d.result.length - 10) + ' 行</p>';
                }

                // 错误
                if (d.error) {
                    html += '<div class="error">❌ ' + escapeHtml(d.error) + '</div>';
                }

                content.innerHTML = html;
            }

            async function loadMetrics() {
                try {
                    const res = await fetch('/metrics');
                    const data = await res.json();
                    const summary = data.summary || {};
                    const content = document.getElementById('metricsContent');
                    content.innerHTML = '<div class="metrics-grid">' +
                        '<div class="metric-card"><div class="value">' + (summary.total_queries || 0) + '</div><div class="label">总查询数</div></div>' +
                        '<div class="metric-card"><div class="value">' + (summary.first_attempt_success_rate || 0) + '%</div><div class="label">一次命中率</div></div>' +
                        '<div class="metric-card"><div class="value">' + (summary.sql_execution_success_rate || 0) + '%</div><div class="label">SQL成功率</div></div>' +
                        '<div class="metric-card"><div class="value">' + (summary.avg_response_time_ms || 0) + 'ms</div><div class="label">平均响应</div></div>' +
                    '</div>';
                } catch (e) {
                    document.getElementById('metricsContent').innerHTML = '<div class="error">加载失败</div>';
                }
            }

            function clearResult() {
                document.getElementById('resultCard').style.display = 'none';
                document.getElementById('resultContent').innerHTML = '';
            }

            function escapeHtml(s) {
                if (!s) return '';
                return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
            }

            // 自动加载
            loadMetrics();
        </script>
    </body>
    </html>
    """)
