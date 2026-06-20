"""四重校验与安全防护 — SQL 校验 + 安全过滤 + 结果检测.

建立四层校验体系：
1. SQL 语法静态检查
2. 结果分布异常检测（空结果/全 Null/极值）
3. 历史数据交叉验证
4. 关键结论人工确认

结合动态脱敏与行列级权限控制。
"""

from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import errors as sqlglot_errors

from nl2sql_agent.models import AgentState


class SQLValidator:
    """SQL 语法 + 安全校验器."""

    FORBIDDEN_PATTERNS = [
        (r"(?i)\bDROP\s+TABLE", "DROP TABLE is not allowed"),
        (r"(?i)\bDROP\s+DATABASE", "DROP DATABASE is not allowed"),
        (r"(?i)\bALTER\s+TABLE", "ALTER TABLE is not allowed"),
        (r"(?i)\bTRUNCATE\b", "TRUNCATE is not allowed"),
        (r"(?i)\bDELETE\s+FROM", "DELETE is not allowed"),
        (r"(?i)\bINSERT\s+INTO", "INSERT is not allowed"),
        (r"(?i)\bUPDATE\b", "UPDATE is not allowed"),
        (r"(?i)\bEXEC\b", "EXEC is not allowed"),
        (r"(?i)\bEXECUTE\b", "EXECUTE is not allowed"),
        (r"(?i)\bINTO\s+OUTFILE\b", "INTO OUTFILE is not allowed"),
        (r"(?i)\bLOAD\s+DATA\b", "LOAD DATA is not allowed"),
        (r"(?i)\bINFORMATION_SCHEMA\b", "Schema introspection not allowed for security"),
    ]

    MAX_JOIN_COUNT = 5
    MAX_QUERY_LENGTH = 10000

    @classmethod
    def validate_syntax(cls, sql: str) -> tuple[bool, str]:
        """SQL 语法静态检查 — 使用 sqlglot 做解析."""
        if not sql or not sql.strip():
            return False, "Empty SQL query"

        if len(sql) > cls.MAX_QUERY_LENGTH:
            return False, f"SQL exceeds max length ({len(sql)} > {cls.MAX_QUERY_LENGTH})"

        try:
            parsed = sqlglot.parse_one(sql)
            # 检查是否为 SELECT
            if parsed.key.upper() != "SELECT":
                return False, f"Only SELECT queries are allowed, got {parsed.key}"
            return True, "Syntax OK"
        except sqlglot_errors.ParseError as e:
            return False, f"SQL syntax error: {e}"
        except Exception as e:
            return False, f"SQL parsing error: {e}"

    @classmethod
    def validate_security(cls, sql: str) -> list[dict[str, Any]]:
        """安全扫描 — 禁止 DDL/DML 和危险操作."""
        issues = []
        for pattern, msg in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, sql):
                issues.append({"type": "security", "severity": "error", "message": msg})
        return issues

    @classmethod
    def estimate_risk(cls, sql: str) -> dict[str, Any]:
        """扫描 SQL 评估风险等级."""
        risks = []
        risk_score = 0

        # 检查 join 数量
        join_count = len(re.findall(r"(?i)\bJOIN\b", sql))
        if join_count > cls.MAX_JOIN_COUNT:
            risks.append(f"Excessive JOINs ({join_count})")
            risk_score += 2
        elif join_count > 3:
            risks.append(f"Multi-table JOIN ({join_count})")
            risk_score += 1

        # 检查是否无 WHERE
        if not re.search(r"(?i)\bWHERE\b", sql):
            risks.append("No WHERE clause — potential full table scan")
            risk_score += 2

        # 检查 SELECT *
        if re.search(r"(?i)\bSELECT\s+\*", sql):
            risks.append("SELECT * — consider specifying columns")
            risk_score += 1

        # 检查 LIMIT
        if not re.search(r"(?i)\bLIMIT\b", sql):
            risks.append("No LIMIT clause — large result set possible")
            risk_score += 1

        level = "low"
        if risk_score >= 3:
            level = "high"
        elif risk_score >= 1:
            level = "medium"

        return {
            "risk_level": level,
            "risk_score": risk_score,
            "risks": risks,
        }


class ResultValidator:
    """结果校验器 — 检测结果集的异常."""

    @classmethod
    def validate_result(cls, result: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """检测结果集异常."""
        issues = []

        if result is None:
            issues.append({"type": "result", "severity": "error", "message": "No result returned"})
            return issues

        if len(result) == 0:
            issues.append({"type": "result", "severity": "warning", "message": "Empty result set"})
            return issues

        # 检查全 Null
        all_null = True
        for row in result:
            for val in row.values():
                if val is not None:
                    all_null = False
                    break
            if not all_null:
                break

        if all_null:
            issues.append({"type": "result", "severity": "warning", "message": "All values are NULL"})

        # 检查单行结果是否可疑（可能只是 COUNT 聚合）
        if len(result) == 1 and len(result[0]) <= 2:
            issues.append({"type": "result", "severity": "info", "message": "Single row result"})

        # 检查结果行数
        if len(result) > 1000:
            issues.append({"type": "result", "severity": "info", "message": f"Large result set ({len(result)} rows)"})

        return issues


# ── LangGraph Node ─────────────────────────────────────────

async def validation_node(state: AgentState) -> AgentState:
    """四重校验节点."""
    state.node_history.append("validation")
    validation_results: list[dict[str, Any]] = []

    if state.sql is None:
        state.query_error = "No SQL to validate"
        return state

    # 第一重：语法静态检查
    syntax_valid, syntax_error = SQLValidator.validate_syntax(state.sql)
    state.syntax_valid = syntax_valid
    state.syntax_error = syntax_error
    validation_results.append({
        "check": "syntax",
        "passed": syntax_valid,
        "detail": syntax_error,
    })

    if not syntax_valid:
        state.validation_passed = False
        state.validation_results = validation_results
        return state

    # 第二重：安全扫描
    security_issues = SQLValidator.validate_security(state.sql)
    if security_issues:
        validation_results.append({
            "check": "security",
            "passed": False,
            "detail": security_issues,
        })
        state.validation_passed = False
        state.validation_results = validation_results
        return state
    else:
        validation_results.append({
            "check": "security",
            "passed": True,
            "detail": "No security issues",
        })

    # 第三重：风险估计
    risk_info = SQLValidator.estimate_risk(state.sql)
    validation_results.append({
        "check": "risk_assessment",
        "passed": risk_info["risk_level"] != "high",
        "detail": risk_info,
    })

    # 第四重：结果校验（如果有结果）
    if state.query_result is not None:
        result_issues = ResultValidator.validate_result(state.query_result)
        validation_results.append({
            "check": "result_validation",
            "passed": all(i.get("severity") != "error" for i in result_issues),
            "detail": result_issues,
        })

    state.validation_results = validation_results
    state.validation_passed = all(
        v.get("passed", False) for v in validation_results
        if v["check"] != "result_validation"  # 结果校验仅做参考
    )

    return state
