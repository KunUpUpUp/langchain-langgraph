"""主题护栏 - 限制 AI 只回答特定领域的问题"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json


# ========== 1. 主题分类器 ==========
def classify_topic(user_input: str) -> dict:
    """使用 LLM 判断用户输入属于哪个主题"""
    llm = ChatOpenAI(model="qwen-plus")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个主题分类器。将用户输入分类到以下主题之一：
- programming: 编程、软件开发相关
- devops: 运维、部署、CI/CD 相关
- general_tech: 其他技术问题
- off_topic: 非技术话题（闲聊、政治、娱乐等）

只回复 JSON：{{"topic": "主题名", "confidence": 0.0-1.0}}"""),
        ("user", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"input": user_input})

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"topic": "off_topic", "confidence": 0.5}


# ========== 2. 各主题的专用处理链 ==========
def build_programming_chain():
    llm = ChatOpenAI(model="qwen-plus")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个资深编程专家。请用清晰的代码示例和解释回答问题。"),
        ("user", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


def build_devops_chain():
    llm = ChatOpenAI(model="qwen-plus")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 DevOps 专家。请从运维和部署的角度回答问题。"),
        ("user", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


def build_general_tech_chain():
    llm = ChatOpenAI(model="qwen-plus")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个技术顾问。请简洁地回答技术相关问题。"),
        ("user", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


# ========== 3. 带主题护栏的路由链 ==========
ALLOWED_TOPICS = {"programming", "devops", "general_tech"}


def chat_with_topic_guard(user_input: str):
    """带主题护栏的对话入口"""
    print(f"用户: {user_input}")

    # 主题分类
    classification = classify_topic(user_input)
    topic = classification["topic"]
    confidence = classification.get("confidence", 0)
    print(f"[分类] 主题={topic}, 置信度={confidence}")

    # 主题护栏：拒绝非技术话题
    if topic not in ALLOWED_TOPICS:
        print("🛡️ 抱歉，我是一个技术助手，只能回答技术相关的问题。")
        return

    # 路由到对应的专用链
    chains = {
        "programming": build_programming_chain(),
        "devops": build_devops_chain(),
        "general_tech": build_general_tech_chain(),
    }

    chain = chains[topic]
    response = chain.invoke({"input": user_input})
    print(f"助手 [{topic}]: {response}")


# ========== 4. 演示 ==========
if __name__ == "__main__":
    # 编程问题 → 路由到编程链
    chat_with_topic_guard("Python 中装饰器的原理是什么？")
    print("\n" + "=" * 50 + "\n")

    # DevOps 问题 → 路由到运维链
    chat_with_topic_guard("如何用 Docker Compose 部署微服务？")
    print("\n" + "=" * 50 + "\n")

    # 非技术话题 → 被护栏拦截
    chat_with_topic_guard("今天晚上吃什么好？")
