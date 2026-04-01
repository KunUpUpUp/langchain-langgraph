"""Token 限制记忆 - 按 token 数截断，比按轮数更精确"""
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

MAX_TOKENS = 500  # 历史消息最多占 500 token


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约1.5字符/token，英文约4字符/token）"""
    return len(text) // 2


def trim_by_tokens(messages: list, max_tokens: int) -> list:
    """从最新消息往前保留，直到超过 token 限制"""
    system = [m for m in messages if m["role"] == "system"]
    history = [m for m in messages if m["role"] != "system"]

    kept = []
    total = 0
    # 从后往前遍历，保留最新的
    for msg in reversed(history):
        tokens = estimate_tokens(msg["content"])
        if total + tokens > max_tokens:
            break
        kept.insert(0, msg)
        total += tokens

    return system + kept


def chat():
    messages = [{"role": "system", "content": "你是一个助手"}]
    print(f"【Token限制记忆】历史消息最多 {MAX_TOKENS} token，输入 quit 退出\n")

    while True:
        user_input = input("你: ").strip()
        if user_input == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        trimmed = trim_by_tokens(messages, MAX_TOKENS)

        response = client.chat.completions.create(
            model="qwen-plus", messages=trimmed
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"助手: {reply}")
        print(f"  [总消息数: {len(messages)}, 发送给LLM: {len(trimmed)}]\n")


if __name__ == "__main__":
    chat()
