"""滑动窗口记忆 - 只保留最近 N 轮对话"""
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

MAX_ROUNDS = 3  # 只保留最近3轮


def sliding_window(messages: list, max_rounds: int) -> list:
    """保留 system + 最近 N 轮（每轮 = 1条user + 1条assistant）"""
    system = [m for m in messages if m["role"] == "system"]
    history = [m for m in messages if m["role"] != "system"]
    # 每轮2条消息，保留最近 max_rounds 轮
    keep = max_rounds * 2
    return system + history[-keep:]


def chat():
    messages = [{"role": "system", "content": "你是一个助手"}]
    print(f"【滑动窗口记忆】只保留最近 {MAX_ROUNDS} 轮对话，输入 quit 退出\n")

    while True:
        user_input = input("你: ").strip()
        if user_input == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        # 截取窗口内的消息发给 LLM
        windowed = sliding_window(messages, MAX_ROUNDS)

        response = client.chat.completions.create(
            model="qwen-plus", messages=windowed
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"助手: {reply}")
        print(f"  [总消息数: {len(messages)}, 发送给LLM: {len(windowed)}]\n")


if __name__ == "__main__":
    chat()
