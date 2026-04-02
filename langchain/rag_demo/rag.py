"""
RAG 示例 - 检索增强生成

流程：
1. 加载知识库文档，切分成小块
2. 用 embedding 模型把文本块转成向量
3. 用户提问时，检索最相关的文本块
4. 把检索到的内容 + 用户问题一起发给 LLM 回答
"""
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from sentence_transformers import SentenceTransformer

client = OpenAI()
# 本地 embedding 模型，首次运行会自动下载（约 100MB）
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


# ========== 工具定义 ==========
def get_current_date() -> str:
    """获取当前日期和时间"""
    from datetime import datetime
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{now.strftime('%Y年%m月%d日 %H:%M:%S')} {weekdays[now.weekday()]}"


TOOLS_MAP = {"get_current_date": get_current_date}

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ========== 1. 加载并切分文档 ==========
def load_documents(knowledge_dir: Path) -> list[dict]:
    """加载所有 md 文件，按段落切分"""
    chunks = []
    for md_file in knowledge_dir.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # 按 ## 标题切分成段落
        sections = content.split("\n## ")
        for section in sections:
            text = section.strip()
            # 去掉开头的 # 标题行
            if text.startswith("# "):
                continue
            chunks.append({
                "text": text,
                "source": md_file.name,
            })
    return chunks


# ========== 2. 文本转向量（本地 Embedding） ==========
def get_embeddings(texts: list[str]) -> list[list[float]]:
    """用本地模型把文本转成向量，完全免费"""
    return embedding_model.encode(texts).tolist()


# ========== 3. 向量相似度检索 ==========
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度"""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query: str, chunks: list[dict], chunk_embeddings: list, top_k: int = 2) -> list[dict]:
    """检索与问题最相关的 top_k 个文本块"""
    query_embedding = get_embeddings([query])[0]

    scored = []
    for i, chunk in enumerate(chunks):
        score = cosine_similarity(query_embedding, chunk_embeddings[i])
        scored.append({"text": chunk["text"], "source": chunk["source"], "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ========== 4. RAG 问答 ==========
def rag_chat(query: str, chunks: list[dict], chunk_embeddings: list, history: list[dict]) -> str:
    """检索相关文档 + LLM 生成回答（带记忆）"""
    # 检索
    results = search(query, chunks, chunk_embeddings)

    print("  [检索到的相关文档]")
    for r in results:
        print(f"    - {r['source']} (相似度: {r['score']:.3f})")

    # 拼接检索到的内容作为上下文
    context = "\n\n".join(f"[来源: {r['source']}]\n{r['text']}" for r in results)

    # 发给 LLM（带工具 + 历史记忆）
    messages = [
        {
            "role": "system",
            "content": f"根据以下参考资料回答用户问题。如果资料中没有相关信息，请说明。\n\n参考资料：\n{context}",
        },
    ]
    # 拼接历史对话（记忆）
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    while True:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=OPENAI_TOOLS,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            print(f"  [调用工具: {name}]")
            result = TOOLS_MAP[name]()
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


# ========== 主程序 ==========
if __name__ == "__main__":
    # 1. 加载知识库
    print("加载知识库...")
    chunks = load_documents(KNOWLEDGE_DIR)
    print(f"  切分为 {len(chunks)} 个文本块\n")

    # 2. 构建向量索引
    print("构建向量索引...")
    chunk_embeddings = get_embeddings([c["text"] for c in chunks])
    print("  完成\n")

    # 3. 交互问答
    history = []  # 短期记忆
    print("【RAG 问答】输入问题，输入 quit 退出\n")
    while True:
        query = input("你: ").strip()
        if query == "quit":
            break

        answer = rag_chat(query, chunks, chunk_embeddings, history)

        # 追加到记忆
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        print(f"\n助手: {answer}")
        print(f"  [记忆轮数: {len(history) // 2}]\n")
