"""NL2SQL Data Agent — 入口."""

import argparse
import asyncio
import os
import sys

import uvicorn


def main():
    """CLI 入口."""
    parser = argparse.ArgumentParser(description="NL2SQL Data Agent")
    parser.add_argument("--web", action="store_true", help="启动 Web 服务")
    parser.add_argument("--host", default="0.0.0.0", help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="Web 服务端口")
    parser.add_argument("--demo", action="store_true", help="运行演示脚本")
    parser.add_argument("--query", type=str, help="单次查询（非交互）")

    args = parser.parse_args()

    if args.web:
        print(f"🚀 启动 Web 服务: http://{args.host}:{args.port}")
        print(f"   API: http://{args.host}:{args.port}/query")
        print(f"   指标: http://{args.host}:{args.port}/metrics")
        uvicorn.run(
            "nl2sql_agent.web:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
    elif args.demo:
        from nl2sql_agent.examples.demo import main as demo_main
        asyncio.run(demo_main())
    elif args.query:
        async def run_single():
            from nl2sql_agent.graph import run_agent, build_data_agent
            from nl2sql_agent.semantic import create_default_registry
            from nl2sql_agent.sql_gen import SQLExecutor
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "qwen-plus"),
                temperature=0.1,
            )
            registry = create_default_registry()
            executor = SQLExecutor()

            # 默认 demo 数据
            executor.set_demo_data([
                {"date": "2024-01-01", "gmv": 1250000},
                {"date": "2024-01-02", "gmv": 1180000},
            ])

            graph, _, _ = build_data_agent(llm, registry, executor)
            output = await run_agent(graph, args.query)

            print(f"📍 路径: {' → '.join(output.node_path)}")
            print(f"⏱ 耗时: {output.execution_time_ms:.0f}ms")
            if output.sql:
                print(f"\n🔍 SQL:\n{output.sql}")
            if output.result:
                import json
                print(f"\n📊 结果:\n{json.dumps(output.result, ensure_ascii=False, indent=2)}")
            if output.error:
                print(f"\n❌ 错误: {output.error}")

        asyncio.run(run_single())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
