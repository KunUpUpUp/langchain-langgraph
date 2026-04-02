"""Memory 示例 - 短期记忆 + 长期记忆"""
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()
MEMORY_FILE = Path(__file__).parent / "long_term_memory.json"


# ========== 长期记忆：持久化到文件（生产中用数据库） ==========
def load_long_term_memory() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {"user_preferences": {}, "facts": []}


def save_long_term_memory(memory: dict):
    MEMORY_FILE.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_facts(assistant_msg: str, user_msg: str, memory: dict):
    """让 LLM 从对话中提取值得记住的信息"""
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": '从以下对话中提取用户的个人偏好或重要信息。返回JSON格式：{"preferences": {"key": "value"}, "facts": ["fact1"]}。如果没有值得记住的信息，返回空：{"preferences": {}, "facts": []}',
            },
            {"role": "user", "content": f"用户说: {user_msg}\n助手回复: {assistant_msg}"},
        ],
    )
    try:
        extracted = json.loads(response.choices[0].message.content)
        memory["user_preferences"].update(extracted.get("preferences", {}))
        for fact in extracted.get("facts", []):
            if fact not in memory["facts"]:
                memory["facts"].append(fact)
        save_long_term_memory(memory)
    except json.JSONDecodeError:
        pass


# ========== 对话主循环 ==========
def chat():
    # 短期记忆：当前对话的 messages 列表
    short_term = []

    # 加载长期记忆
    long_term = load_long_term_memory()

    # 把长期记忆注入 system prompt
    system_prompt = "你是一个有记忆的助手。"
    if long_term["user_preferences"] or long_term["facts"]:
        system_prompt += "\n\n你记得关于用户的以下信息："
        if long_term["user_preferences"]:
            system_prompt += f"\n偏好: {json.dumps(long_term['user_preferences'], ensure_ascii=False)}"
        if long_term["facts"]:
            system_prompt += f"\n事实: {', '.join(long_term['facts'])}"
        print(f"[已加载长期记忆: {len(long_term['user_preferences'])} 条偏好, {len(long_term['facts'])} 条事实]\n")

    short_term.append({"role": "system", "content": system_prompt})

    print("开始对话（输入 quit 退出）\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "quit":
            break

        # 短期记忆：追加用户消息
        short_term.append({"role": "user", "content": user_input})

        # 调用 LLM（带完整短期记忆）
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=short_term,
        )

        assistant_msg = response.choices[0].message.content

        # 短期记忆：追加助手回复
        short_term.append({"role": "assistant", "content": assistant_msg})

        print(f"助手: {assistant_msg}\n")

        # 异步提取长期记忆（从对话中提取值得记住的信息）
        extract_facts(assistant_msg, user_input, long_term)

    print(f"\n[长期记忆已保存到 {MEMORY_FILE}]")


if __name__ == "__main__":
    chat()
