from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def search(query: str) -> str:
    """获取天气信息"""
    # if not query.strip():
    raise Exception("用户需要给出具体城市")
    return f"结果：{query}"

# @wrap_tool_call
# def handle_tool_errors(request, handler):
#     """使用自定义消息处理工具执行错误。"""
#     try:
#         return handler(request)
#     except Exception as e:
#         # 向模型返回自定义错误消息
#         return ToolMessage(
#             content=f"工具错误：请检查您的输入并重试。({str(e)})",
#             tool_call_id=request.tool_call["id"]
#         )

agent = create_agent(
    model="openai:qwen-plus",
    tools=[search]
    # tools=[search],
    # middleware=[handle_tool_errors]
)

response = agent.invoke(
    # {"messages": [{"role": "user", "content": "旧金山天气如何？"}]}
    {"messages": [{"role": "user", "content": "今天天气如何？"}]}
)

print(response["messages"])