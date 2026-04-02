"""摘要记忆 - 早期对话压缩成摘要，保留关键信息又省 token"""
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

SUMMARY_THRESHOLD = 6  # 超过6条消息时触发摘要


def summarize(messages: list) -> str:
    """让 LLM 把历史对话压缩成一段摘要"""
    conversation = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system"
    )
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "请用2-3句话总结以下对话的关键信息，保留重要事实和用户偏好："},
            {"role": "user", "content": conversation},
        ],
    )
    return response.choices[0].message.content


def chat():
    messages = [{"role": "system", "content": "你是一个助手"}]
    summary = ""
    print(f"【摘要记忆】超过 {SUMMARY_THRESHOLD} 条消息时自动摘要，输入 quit 退出\n")

    while True:
        user_input = input("你: ").strip()
        if user_input == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        # 消息过多时，把早期对话压缩成摘要
        history = [m for m in messages if m["role"] != "system"]
        if len(history) > SUMMARY_THRESHOLD:
            # 摘要前面的对话，保留最近2轮
            to_summarize = history[:-4]
            summary = summarize(to_summarize)
            # 只保留 system + 最近2轮
            messages = [m for m in messages if m["role"] == "system"] + history[-4:]
            print(f"  [触发摘要压缩，摘要: {summary[:50]}...]\n")

        # 构建发送给 LLM 的消息
        send = []
        system_content = messages[0]["content"]
        if summary:
            system_content += f"\n\n之前的对话摘要：{summary}"
        send.append({"role": "system", "content": system_content})
        send.extend([m for m in messages if m["role"] != "system"])

        response = client.chat.completions.create(
            model="qwen-plus", messages=send
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"助手: {reply}")
        print(f"  [当前消息数: {len(messages)}, 有摘要: {'是' if summary else '否'}]\n")


if __name__ == "__main__":
    chat()
