"""完整历史记忆 - 保留所有对话，最简单但最费 token"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()


def chat():
    messages = [{"role": "system", "content": "你是一个助手"}]
    print("【完整历史记忆】所有对话全部保留，输入 quit 退出\n")

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

        print(f"助手: {reply}")
        print(f"  [当前消息数: {len(messages)}]\n")


if __name__ == "__main__":
    chat()
