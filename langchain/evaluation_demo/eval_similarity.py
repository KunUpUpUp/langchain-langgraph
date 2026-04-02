"""
Evaluation 示例 3 - 语义相似度评估（Embedding-based）

不依赖 LLM 评判，而是用 Embedding 向量计算语义相似度。
速度快、成本低，适合大规模自动化评估。

流程：
1. 准备参考答案和模型回答
2. 用 Embedding 模型将两者转为向量
3. 计算余弦相似度作为评分
4. 设定阈值判断是否通过
"""
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-v3"
PASS_THRESHOLD = 0.75  # 相似度阈值


# ========== 1. 测试数据 ==========
TEST_CASES = [
    {
        "question": "什么是递归？",
        "reference": "递归是函数直接或间接调用自身的编程技巧，需要有终止条件防止无限循环。",
        "prediction": "递归就是一个函数在执行过程中调用它自己，必须设置基准条件来结束递归。",
    },
    {
        "question": "什么是递归？",
        "reference": "递归是函数直接或间接调用自身的编程技巧，需要有终止条件防止无限循环。",
        "prediction": "今天天气真不错，适合出去散步。",
    },
    {
        "question": "解释 RESTful API",
        "reference": "RESTful API 是基于 HTTP 协议的接口设计风格，使用 URL 表示资源，用 HTTP 方法表示操作。",
        "prediction": "REST API 是一种 Web 接口规范，通过 HTTP 动词（GET/POST/PUT/DELETE）对 URL 资源进行 CRUD 操作。",
    },
    {
        "question": "解释 RESTful API",
        "reference": "RESTful API 是基于 HTTP 协议的接口设计风格，使用 URL 表示资源，用 HTTP 方法表示操作。",
        "prediction": "API 就是应用程序接口，可以用来连接不同的软件系统。",
    },
]


# ========== 2. Embedding + 相似度计算 ==========
def get_embedding(text: str) -> list[float]:
    """获取文本的 embedding 向量"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度"""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ========== 3. 评估函数 ==========
def evaluate_similarity(reference: str, prediction: str) -> dict:
    """计算参考答案和模型回答的语义相似度"""
    ref_emb = get_embedding(reference)
    pred_emb = get_embedding(prediction)
    score = cosine_similarity(ref_emb, pred_emb)
    return {
        "score": score,
        "passed": score >= PASS_THRESHOLD,
    }


def run_evaluation(test_cases: list[dict]) -> list[dict]:
    """批量评估"""
    results = []
    for case in test_cases:
        eval_result = evaluate_similarity(case["reference"], case["prediction"])
        results.append({**case, **eval_result})
    return results


# ========== 4. 报告 ==========
def print_report(results: list[dict]):
    """打印评估报告"""
    print("=" * 60)
    print("📊 语义相似度评估报告")
    print(f"   通过阈值: {PASS_THRESHOLD}")
    print("=" * 60)

    passed_count = 0
    for i, r in enumerate(results, 1):
        status = "✅ 通过" if r["passed"] else "❌ 未通过"
        if r["passed"]:
            passed_count += 1

        print(f"\n--- 第 {i} 题 ---")
        print(f"问题：{r['question']}")
        print(f"参考：{r['reference'][:50]}...")
        print(f"回答：{r['prediction'][:50]}...")
        print(f"相似度：{r['score']:.4f}  {status}")

    print(f"\n{'=' * 60}")
    print(f"通过率：{passed_count}/{len(results)} ({passed_count/len(results)*100:.0f}%)")


# ========== 主程序 ==========
if __name__ == "__main__":
    print("开始语义相似度评估...\n")
    results = run_evaluation(TEST_CASES)
    print_report(results)
