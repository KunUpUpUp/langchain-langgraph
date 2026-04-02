"""
Planning 示例 - LLM 任务规划与执行

流程：
1. 用户输入一个复杂目标
2. LLM 将目标拆解为多个子任务（Planning）
3. 按顺序执行每个子任务，调用工具完成
4. 汇总所有子任务结果，生成最终回答

这是 Agent 中最核心的能力之一：把大问题拆成小步骤，逐步解决。
"""
import json
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()
MODEL = "qwen-plus"


# ========== 工具定义 ==========
def search_web(query: str) -> str:
    """模拟搜索引擎"""
    fake_results = {
        "Python 最新版本": "Python 3.13，于 2024 年 10 月发布，新增了实验性的自由线程模式和 JIT 编译器。",
        "Rust 最新版本": "Rust 1.83，改进了异步编程和编译速度。",
        "Go 最新版本": "Go 1.23，增强了泛型支持和标准库迭代器。",
    }
    for key, val in fake_results.items():
        if key in query or any(k in query for k in key.split()):
            return val
    return f"搜索结果：关于 '{query}' 的最新信息暂未找到。"


def calculate(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "不支持的表达式"
    except Exception as e:
        return f"计算错误: {e}"


def summarize_text(text: str) -> str:
    """用 LLM 总结文本"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "请用一句话总结以下内容："},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


TOOLS_MAP = {
    "search_web": search_web,
    "calculate": calculate,
    "summarize_text": summarize_text,
}

OPENAI_TOOLS = [
    {"type": "function", "function": {"name": "search_web", "description": "搜索网络获取信息", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "summarize_text", "description": "总结一段文本", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "需要总结的文本"}}, "required": ["text"]}}},
]


# ========== 1. Planning：让 LLM 拆解任务 ==========
def create_plan(goal: str) -> list[dict]:
    """让 LLM 把一个复杂目标拆解为有序的子任务列表"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """你是一个任务规划专家。用户会给你一个目标，你需要把它拆解为 2-5 个具体的子任务。

返回 JSON 数组格式，每个子任务包含：
- step: 步骤编号
- task: 任务描述
- tool: 需要使用的工具（search_web / calculate / summarize_text / none）
- input: 工具的输入参数

可用工具：
- search_web: 搜索网络信息，输入为搜索关键词
- calculate: 计算数学表达式
- summarize_text: 总结文本
- none: 不需要工具，由 LLM 直接回答

只返回 JSON 数组，不要其他内容。""",
            },
            {"role": "user", "content": goal},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()
    # 处理可能的 markdown 代码块包裹
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(content)


# ========== 2. 执行单个子任务 ==========
def execute_step(step: dict, context: str) -> str:
    """执行一个子任务，返回结果"""
    tool_name = step.get("tool", "none")
    tool_input = step.get("input", "")

    if tool_name in TOOLS_MAP:
        # 调用工具
        result = TOOLS_MAP[tool_name](tool_input)
        print(f"    [工具 {tool_name}] 输入: {tool_input}")
        print(f"    [工具 {tool_name}] 输出: {result}")
        return result
    else:
        # 不需要工具，让 LLM 基于已有上下文回答
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"根据以下已知信息完成任务。\n\n已知信息：\n{context}",
                },
                {"role": "user", "content": step["task"]},
            ],
        )
        result = response.choices[0].message.content
        print(f"    [LLM 推理] {result[:100]}...")
        return result


# ========== 3. Plan-and-Execute 主流程 ==========
def plan_and_execute(goal: str) -> str:
    """完整的 Planning 流程：规划 → 逐步执行 → 汇总"""
    print(f"目标: {goal}\n")

    # Step 1: 生成计划
    print("=" * 50)
    print("📋 第一阶段：生成计划")
    print("=" * 50)
    plan = create_plan(goal)

    for step in plan:
        print(f"  步骤 {step['step']}: {step['task']} [工具: {step.get('tool', 'none')}]")
    print()

    # Step 2: 逐步执行
    print("=" * 50)
    print("⚙️ 第二阶段：逐步执行")
    print("=" * 50)
    results = []
    context = ""  # 累积上下文，后续步骤可以引用前面的结果

    for step in plan:
        print(f"\n  >> 步骤 {step['step']}: {step['task']}")
        result = execute_step(step, context)
        results.append({"step": step["step"], "task": step["task"], "result": result})
        context += f"\n步骤{step['step']}({step['task']})的结果：{result}"

    # Step 3: 汇总
    print(f"\n{'=' * 50}")
    print("📝 第三阶段：汇总结果")
    print("=" * 50)

    summary_input = f"用户目标：{goal}\n\n执行结果：\n"
    for r in results:
        summary_input += f"步骤{r['step']}({r['task']}): {r['result']}\n"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个助手。根据以下任务执行结果，给用户一个完整、清晰的最终回答。",
            },
            {"role": "user", "content": summary_input},
        ],
    )
    return response.choices[0].message.content


# ========== 主程序 ==========
if __name__ == "__main__":
    # 示例 1：多步骤信息收集
    print("【示例 1】多步骤信息收集\n")
    answer = plan_and_execute("帮我对比 Python、Rust、Go 三种语言的最新版本，并总结各自的亮点")
    print(f"\n最终回答:\n{answer}")

    print("\n" + "=" * 70 + "\n")

    # 示例 2：计算 + 推理
    print("【示例 2】计算 + 推理\n")
    answer = plan_and_execute("一个程序员月薪 25000，每月房租 5000，生活费 3000，每月能存多少钱？一年能存多少？")
    print(f"\n最终回答:\n{answer}")
