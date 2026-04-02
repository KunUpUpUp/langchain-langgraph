"""
Evaluation 示例 2 - QA 问答评估（带参考答案）

用 LLM 评估模型回答与标准答案的一致性。
适用于有明确正确答案的场景（如知识问答、考试题）。

流程：
1. 准备测试集（问题 + 参考答案）
2. 让被测模型生成回答
3. 用评估模型对比参考答案和生成回答
4. 统计评估结果
"""
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()
MODEL = "qwen-plus"

# ========== 1. 测试数据集 ==========
TEST_CASES = [
    {
        "question": "Python 中 `is` 和 `==` 有什么区别？",
        "reference": "== 比较值是否相等，is 比较是否是同一个对象（内存地址相同）。",
    },
    {
        "question": "什么是 Git 的 rebase？",
        "reference": "rebase 是将一个分支的提交重新应用到另一个分支的基础上，使提交历史更线性。",
    },
    {
        "question": "Docker 容器和虚拟机有什么区别？",
        "reference": "容器共享宿主机内核，更轻量快速；虚拟机有独立内核，隔离性更强但开销更大。",
    },
    {
        "question": "HTTP GET 和 POST 的区别？",
        "reference": "GET 用于获取资源，参数在 URL 中，幂等；POST 用于提交数据，参数在请求体中，非幂等。",
    },
]


# ========== 2. 生成模型回答 ==========
def generate_answer(question: str) -> str:
    """让被测模型回答问题"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "请简洁准确地回答技术问题，控制在 2-3 句话以内。"},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ========== 3. 评估回答质量 ==========
def evaluate_answer(question: str, reference: str, prediction: str) -> dict:
    """对比参考答案和模型回答，判断是否正确"""
    prompt = f"""请作为评估专家，判断 AI 的回答是否与参考答案在语义上一致。

问题：{question}
参考答案：{reference}
AI 回答：{prediction}

请按以下格式输出：
判定：[正确 / 部分正确 / 错误]
得分：[1.0 / 0.5 / 0.0]
分析：[简要说明]"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return {
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "eval_output": response.choices[0].message.content,
    }


# ========== 4. 批量评估 + 统计 ==========
def run_evaluation(test_cases: list[dict]) -> list[dict]:
    """对整个测试集进行评估"""
    results = []
    for case in test_cases:
        print(f"  评估中: {case['question'][:30]}...")
        prediction = generate_answer(case["question"])
        result = evaluate_answer(case["question"], case["reference"], prediction)
        results.append(result)
    return results


def print_report(results: list[dict]):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("📊 QA 评估报告")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        print(f"\n--- 第 {i} 题 ---")
        print(f"问题：{r['question']}")
        print(f"参考：{r['reference']}")
        print(f"回答：{r['prediction'][:80]}...")
        print(f"评估：\n{r['eval_output']}")

    print(f"\n{'=' * 60}")
    print(f"共评估 {len(results)} 道题")


# ========== 主程序 ==========
if __name__ == "__main__":
    print("开始 QA 评估...\n")
    results = run_evaluation(TEST_CASES)
    print_report(results)
