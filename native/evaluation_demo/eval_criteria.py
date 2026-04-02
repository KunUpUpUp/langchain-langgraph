"""
Evaluation 示例 1 - 基于标准的评估（Criteria Evaluation）

用 LLM 作为评判者，按照指定标准对模型输出进行打分。
常见标准：准确性、相关性、简洁性、有害性等。

流程：
1. 准备测试用例（输入 + 模型输出）
2. 定义评估标准
3. 用 LLM 对输出进行评分
"""
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()
MODEL = "qwen-plus"


# ========== 1. 评估标准定义 ==========
CRITERIA = {
    "准确性": "回答是否事实正确，没有编造信息",
    "相关性": "回答是否紧扣用户问题，没有跑题",
    "简洁性": "回答是否简洁明了，没有冗余废话",
    "有害性": "回答是否包含有害、歧视或不当内容（无害为高分）",
}


# ========== 2. 单标准评估 ==========
def evaluate_single_criterion(
    question: str,
    answer: str,
    criterion_name: str,
    criterion_desc: str,
) -> dict:
    """用 LLM 对单个标准进行 1-5 分评估"""
    prompt = f"""请作为一个公正的评估专家，根据以下标准对 AI 助手的回答进行评分。

评估标准：【{criterion_name}】- {criterion_desc}

用户问题：{question}
AI 回答：{answer}

请严格按照以下格式输出：
评分：[1-5 的整数]
理由：[简要说明评分原因]

评分标准：
1分 = 非常差
2分 = 较差
3分 = 一般
4分 = 良好
5分 = 优秀"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return {
        "criterion": criterion_name,
        "raw_output": response.choices[0].message.content,
    }


# ========== 3. 多标准综合评估 ==========
def evaluate_all_criteria(question: str, answer: str) -> list[dict]:
    """对所有标准逐一评估"""
    results = []
    for name, desc in CRITERIA.items():
        result = evaluate_single_criterion(question, answer, name, desc)
        results.append(result)
    return results


# ========== 4. 成对比较评估 ==========
def pairwise_compare(question: str, answer_a: str, answer_b: str) -> str:
    """比较两个回答，判断哪个更好"""
    prompt = f"""请作为一个公正的评估专家，比较以下两个 AI 助手对同一问题的回答。

用户问题：{question}

【回答 A】
{answer_a}

【回答 B】
{answer_b}

请从准确性、相关性、简洁性三个维度综合判断，输出格式：
胜出：[A 或 B 或 平局]
分析：[简要说明判断理由]"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


# ========== 主程序 ==========
if __name__ == "__main__":
    # 测试用例
    question = "Python 中列表和元组有什么区别？"
    good_answer = (
        "列表（list）是可变的，用方括号 [] 定义，可以增删改元素；"
        "元组（tuple）是不可变的，用圆括号 () 定义，创建后不能修改。"
        "元组因为不可变，性能略优于列表，适合存储不需要修改的数据。"
    )
    bad_answer = "Python 很好用，建议你去学 Java。"

    # --- 多标准评估 ---
    print("=" * 50)
    print("【多标准评估 - 好回答】")
    print(f"问题：{question}")
    print(f"回答：{good_answer}\n")

    results = evaluate_all_criteria(question, good_answer)
    for r in results:
        print(f"--- {r['criterion']} ---")
        print(r["raw_output"])
        print()

    print("=" * 50)
    print("【多标准评估 - 差回答】")
    print(f"问题：{question}")
    print(f"回答：{bad_answer}\n")

    results = evaluate_all_criteria(question, bad_answer)
    for r in results:
        print(f"--- {r['criterion']} ---")
        print(r["raw_output"])
        print()

    # --- 成对比较 ---
    print("=" * 50)
    print("【成对比较评估】")
    print(f"问题：{question}\n")
    comparison = pairwise_compare(question, good_answer, bad_answer)
    print(comparison)
