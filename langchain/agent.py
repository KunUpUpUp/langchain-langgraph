"""
完整 Agent 示例 - 集成 Tool + Memory + RAG + Guardrails

使用 LangChain v1 的 create_agent + middleware
"""
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import before_model, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any


# ========== 1. 工具定义 ==========
@tool
def get_current_date() -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{now.strftime('%Y年%m月%d日 %H:%M:%S')} {weekdays[now.weekday()]}"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "晴天 25°C",
        "上海": "多云 22°C",
        "广州": "小雨 28°C",
    }
    return weather_data.get(city, f"{city} 暂无天气数据")


@tool
def search_knowledge(query: str) -> str:
    """搜索知识库获取技术文档信息"""
    knowledge = {
        "docker": "Docker 常用命令：docker build 构建镜像，docker run 运行容器。数据持久化用 Volume。",
        "git": "Git 撤销：git reset --soft HEAD~1 撤销提交保留更改，git stash 临时保存。",
        "python": "Python 技巧：列表推导式 [x**2 for x in range(10)]，字典合并 dict1 | dict2（3.9+）。",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "知识库中未找到相关信息"



# ========== 2. Guardrails（输入护栏，用 before_model middleware） ==========
BLOCKED_TOPICS = ["政治", "赌博", "暴力"]


@before_model(can_jump_to=["end"])
def input_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """在请求到达 LLM 之前检查输入，拦截敏感话题"""
    last_message = state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    for topic in BLOCKED_TOPICS:
        if topic in content:
            return {
                "messages": [AIMessage(content=f"🛡️ 抱歉，我无法回答关于「{topic}」的问题。")],
                "jump_to": "end",
            }
    return None


# ========== 3. 构建 Agent ==========
tools = [get_current_date, get_weather, search_knowledge]

agent = create_agent(
    model="openai:qwen-plus",
    tools=tools,
    system_prompt="你是一个技术助手，可以查询天气、日期和技术知识库。请用中文回答。",
    middleware=[input_guardrail],
)


# ========== 4. 对话循环 ==========
def chat():
    print("【完整 Agent】集成 Tool + Memory + RAG + Guardrails")
    print("输入 quit 退出\n")

    messages = []

    while True:
        user_input = input("你: ").strip()
        if user_input == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        result = agent.invoke({"messages": messages})

        reply = result["messages"][-1].content
        messages = result["messages"]

        print(f"助手: {reply}\n")


if __name__ == "__main__":
    chat()
