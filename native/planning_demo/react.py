"""
ReAct Planning 示例 - 思考-行动-观察 循环

与 plan_and_solve.py 的区别：
- plan_and_solve.py: 先一次性生成完整计划，再逐步执行（Plan-and-Execute）
- react.py: 每一步都重新思考，根据上一步结果决定下一步（ReAct）

ReAct 更灵活，能根据中间结果动态调整策略。
"""
import json
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()
MODEL = "qwen-plus"


# ========== 工具（与 plan_and_solve.py 共用） ==========
def search_web(query: str) -> str:
    """模拟搜索"""
    fake_db = {
        "北京天气": "北京今天晴，气温 15-25°C，空气质量良好。",
        "上海天气": "上海今天多云，气温 18-22°C，有轻度雾霾。",
        "北京美食推荐": "推荐：全聚德烤鸭、炸酱面、豆汁焦圈、涮羊肉。",
        "上海美食推荐": "推荐：小笼包、生煎、红烧肉、蟹粉豆腐。",
    }
    for key, val in fake_db.items():
        if any(k in query for k in key.split()):
            return val
    return f"未找到关于 '{query}' 的信息。"


def calculate(expression: str) -> str:
    """安全计算"""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "不支持的表达式"
    except Exception as e:
        return f"计算错误: {e}"


TOOLS_MAP = {"search_web": search_web, "calculate": calculate}

TOOLS_DESC = """可用工具：
1. search_web(query): 搜索网络信息
2. calculate(expression): 计算数学表达式
3. finish(answer): 给出最终答案，结束任务"""


# ========== ReAct 循环 ==========
def react_agent(goal: str, max_steps: int = 8) -> str:
    """
    ReAct 循环：
    1. Thought（思考）：分析当前状态，决定下一步
    2. Action（行动）：调用工具
    3. Observation（观察）：获取工具返回结果
    重复直到得出最终答案
    """
    print(f"目标: {goal}\n")

    system_prompt = f"""你是一个 ReAct Agent。你需要通过"思考-行动-观察"循环来完成用户的目标。

{TOOLS_DESC}

每次回复必须严格按以下 JSON 格式：
{{
    "thought": "你的思考过程",
    "action": "工具名称（search_web / calculate / finish）",
    "action_input": "工具的输入参数"
}}

规则：
- 每次只执行一个动作
- 根据观察结果决定下一步
- 信息足够时使用 finish 给出最终答案
- 只返回 JSON，不要其他内容"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal},
    ]

    for step in range(1, max_steps + 1):
        print(f"--- 第 {step} 步 ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
        )

        content = response.choices[0].message.content.strip()
        # 处理 markdown 代码块
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            action_data = json.loads(content)
        except json.JSONDecodeError:
            print(f"  [解析失败] {content[:100]}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "请严格按 JSON 格式回复。"})
            continue

        thought = action_data.get("thought", "")
        action = action_data.get("action", "")
        action_input = action_data.get("action_input", "")

        print(f"  💭 思考: {thought}")
        print(f"  🔧 行动: {action}({action_input})")

        # 检查是否结束
        if action == "finish":
            print(f"\n✅ 最终答案: {action_input}")
            return action_input

        # 执行工具
        if action in TOOLS_MAP:
            observation = TOOLS_MAP[action](action_input)
        else:
            observation = f"未知工具: {action}"

        print(f"  👁️ 观察: {observation}")

        # 把这一轮的结果加入对话
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "达到最大步数，未能完成任务。"


# ========== 主程序 ==========
if __name__ == "__main__":
    # 示例：需要多步搜索 + 综合判断的问题
    print("【ReAct 示例】动态规划\n")
    answer = react_agent("我想周末去北京或上海玩，帮我看看哪个城市天气更好，顺便推荐当地美食")
    print(f"\n{'=' * 50}")
    print(f"回答: {answer}")
