from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv


load_dotenv()

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成系统提示。"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有帮助的助手。"

    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术响应。"
    elif user_role == "beginner":
        return f"{base_prompt} 简单解释概念，避免使用行话。"

    return base_prompt

# 没看懂这里为啥传了个类
agent = create_agent(
    model="openai:qwen-plus",
    middleware=[user_role_prompt],
    context_schema=Context
)

# 系统提示将根据上下文动态设置
result = agent.invoke(
    {"messages": [{"role": "user", "content": "解释机器学习"}]},
    context={"user_role": "expert"}
)

print(result["messages"])