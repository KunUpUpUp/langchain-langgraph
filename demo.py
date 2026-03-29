from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city} 天气总是晴朗！"

agent = create_agent(
    model="openai:qwen-plus",
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手",
)

# 执行代理
result = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山天气如何？"}]}
)

print(result["messages"])
# -1 就是 AIMessage 最终结果，content就是它的内容
print(result["messages"][-1].content)
