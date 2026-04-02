"""
完整 Agent 示例 - 集成 Tool + Memory + RAG + Guardrails

用 LangGraph 的 create_react_agent 构建，
不再手写循环，框架自动处理 思考→工具调用→观察 的 ReAct 循环
"""
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


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
    """搜索知识库获取技术文档信息（RAG 简化版）"""
    # 实际中这里接向量数据库检索，这里用简单匹配模拟
    knowledge = {
        "docker": "Docker 常用命令：docker build 构建镜像，docker run 运行容器，docker-compose 编排多容器。数据持久化用 Volume。",
        "git": "Git 撤销：git reset --soft HEAD~1 撤销提交保留更改，git stash 临时保存。SSH 配置：ssh-keygen -t ed25519 生成密钥。",
        "python": "Python 技巧：列表推导式 [x**2 for x in range(10)]，字典合并 dict1 | dict2（3.9+），match-case 模式匹配（3.10+）。",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "知识库中未找到相关信息"


# ========== 2. Guardrails（输入护栏） ==========
BLOCKED_TOPICS = ["政治", "赌博", "暴力"]


def input_guardrail(user_input: str) -> str | None:
    """检查用户输入，返回 None 表示通过，返回字符串表示拦截原因"""
    for topic in BLOCKED_TOPICS:
        if topic in user_input:
            return f"抱歉，我无法回答关于「{topic}」的问题。"
    return None


# ========== 3. 构建 Agent ==========
llm = ChatOpenAI(model="qwen-plus")
tools = [get_current_date, get_weather, search_knowledge]

# Memory：LangGraph 的 MemorySaver 自动管理对话历史
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt="你是一个技术助手，可以查询天气、日期和技术知识库。请用中文回答。",
)


# ========== 4. 对话循环 ==========
def chat():
    # thread_id 标识一个对话，同一个 thread_id 共享记忆
    config = {"configurable": {"thread_id": "user-001"}}

    print("【完整 Agent】集成 Tool + Memory + RAG + Guardrails")
    print("输入 quit 退出\n")

    while True:
        user_input = input("你: ").strip()
        if user_input == "quit":
            break

        # Guardrails 检查
        blocked = input_guardrail(user_input)
        if blocked:
            print(f"🛡️ {blocked}\n")
            continue

        # 调用 Agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # 输出结果
        reply = result["messages"][-1].content
        print(f"助手: {reply}\n")


if __name__ == "__main__":
    chat()
