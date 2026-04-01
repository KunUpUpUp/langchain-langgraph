"""MCP 客户端 - LLM Agent 动态发现并调用工具"""
import asyncio
import json
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def mcp_tools_to_openai_tools(mcp_tools):
    """把 MCP 工具 schema 转成 OpenAI function calling 格式"""
    openai_tools = []
    for tool in mcp_tools.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })
    return openai_tools


async def main():
    python = "/usr/local/project/agent/langchain-langgraph/.venv/bin/python"
    llm = OpenAI()

    # 连接 MCP 服务端
    server_params = StdioServerParameters(command=python, args=["server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. 动态发现工具（不需要知道有什么工具、参数是什么）
            mcp_tools = await session.list_tools()
            openai_tools = mcp_tools_to_openai_tools(mcp_tools)

            print("发现工具:", [t["function"]["name"] for t in openai_tools])

            # 2. 用户提问
            user_input = "北京和上海的天气怎么样？"
            messages = [{"role": "user", "content": user_input}]

            # 3. LLM 根据工具 schema 自己决定调什么、传什么参数
            response = llm.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                tools=openai_tools,
            )

            msg = response.choices[0].message
            messages.append(msg)

            # 4. 如果 LLM 决定调用工具，执行工具调用
            while msg.tool_calls:
                for tool_call in msg.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    print(f"LLM 决定调用: {name}({args})")

                    # 调用 MCP 服务端的工具
                    result = await session.call_tool(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.content[0].text,
                    })

                # 把工具结果返回给 LLM，让它继续思考或生成最终回答
                response = llm.chat.completions.create(
                    model="qwen-plus",
                    messages=messages,
                    tools=openai_tools,
                )
                msg = response.choices[0].message
                messages.append(msg)

            # 5. 最终回答
            print(f"\nLLM 回答: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
