"""长期记忆 - 跨对话持久化，记住用户信息"""
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()
MEMORY_FILE = Path(__file__).parent / "long_term_memory.json"


def load_memory() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {"facts": []}


def save_memory(memory: dict):
    MEMORY_FILE.write_text(
        json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def extract_and_save(user_msg: str, assistant_msg: str, memory: dict):
    """从对话中提取值得长期记住的信息"""
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": '从对话中提取用户的关键信息（姓名、偏好、职业等），返回JSON数组，如 ["用户叫张三", "喜欢Python"]。没有则返回 []',
            },
            {"role": "user", "content": f"用户: {user_msg}\n助手: {assistant_msg}"},
        ],
    )
    try:
        facts = json.loads(response.choices[0].message.content)
        for fact in facts:
            if fact not in memory["facts"]:
                memory["facts"].append(fact)
                print(f"  [记住了: {fact}]")
        save_memory(memory)
    except json.JSONDecodeError:
        pass


def chat():
    memory = load_memory()
    messages = [{"role": "system", "content": "你是一个助手"}]

    if memory["facts"]:
        facts_str = "、".join(memory["facts"])
        messages[0]["content"] += f"\n\n你记得关于用户的信息：{facts_str}"
        print(f"[加载长期记忆: {facts_str}]\n")

    print("【长期记忆】跨对话记住用户信息，输入 quit 退出\n")

    while True:
        user_input = input("你: ").strip()
        if user_input == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="qwen-plus", messages=messages
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"助手: {reply}\n")

        # 提取长期记忆
        extract_and_save(user_input, reply, memory)

    print(f"\n[记忆已保存，下次启动会自动加载]")


if __name__ == "__main__":
    chat()
