"""
RAG Agent —— 使用 LangGraph 实现
- Embedding: sentence-transformers 免费模型（BAAI/bge-small-zh-v1.5，中文友好）
- 向量存储: FAISS（本地索引文件 knowledge_base.index）
- 知识库文档: documents.json（可直接编辑添加文档，程序负责加载和写入）
"""

from typing import TypedDict, Sequence, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from dotenv import load_dotenv
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()

# ==================== 1. 文件路径 ====================

DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_FILE = os.path.join(DIR, "documents.json")  # 原始文档（人工编辑）
FAISS_INDEX_FILE = os.path.join(DIR, "knowledge_base.index")  # FAISS 向量索引
DOCS_META_FILE = os.path.join(DIR, "docs_meta.json")  # 文档文本（与索引一一对应）


# ==================== 2. 文档读写 ====================

def load_documents() -> list[dict]:
    """从 documents.json 加载原始文档"""
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def add_document(text: str):
    """向 documents.json 追加一条新文档"""
    documents = load_documents()
    new_id = max(doc["id"] for doc in documents) + 1 if documents else 1
    documents.append({"id": new_id, "text": text})

    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"✅ 已添加文档 #{new_id}：{text[:50]}...")


# ==================== 3. Embedding + FAISS ====================

embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 全局变量，懒加载
_index = None
_doc_texts = None


def build_knowledge_base():
    """从 documents.json 生成向量嵌入，构建 FAISS 索引并保存"""
    global _index, _doc_texts
    documents = load_documents()
    texts = [doc["text"] for doc in documents]
    embeddings = embed_model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积索引（向量已归一化，内积 = 余弦相似度）
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, FAISS_INDEX_FILE)

    # 保存文档文本，与索引顺序一一对应
    with open(DOCS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    _index = index
    _doc_texts = texts
    print(f"✅ FAISS 索引已构建，共 {len(texts)} 条文档 → knowledge_base.index")


def _load_index():
    """懒加载 FAISS 索引和文档文本"""
    global _index, _doc_texts
    if _index is None:
        _index = faiss.read_index(FAISS_INDEX_FILE)
        with open(DOCS_META_FILE, "r", encoding="utf-8") as f:
            _doc_texts = json.load(f)


def retrieve(query: str, top_k: int = 3) -> list[str]:
    """用 FAISS 检索最相关的文档"""
    _load_index()
    query_vec = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = _index.search(query_vec, top_k)

    results = []
    for idx, val in enumerate(indices[0]):
        if val >= 0 and scores[0][idx] > 0.5:  # FAISS 用 -1 表示无效结果
            results.append(_doc_texts[val])
    return results


# ==================== 4. LLM ====================

model = ChatOpenAI(model="qwen-plus")


# ==================== 5. LangGraph RAG ====================

class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str


def retrieve_node(state: RAGState) -> dict:
    """检索节点：根据用户问题检索相关文档"""
    query = state["messages"][-1].content
    documents = retrieve(query, top_k=3)
    return {"context": "\n\n".join(documents)}


def generate_node(state: RAGState) -> dict:
    """生成节点：结合上下文回答问题"""
    """ 一个好的prompt可以完全给你两种不同结果，用下面的prompt，模型仍然会输出项羽是谁的回答
    
    system_prompt = SystemMessage(content=(
        "你是一个知识问答助手。请根据以下检索到的上下文来回答用户问题。"
        "如果上下文中没有相关信息，请诚实地说你不知道，严禁说出没有相关信息的回答\n\n"
        f"上下文：\n{state['context']}"
    ))

    而用下面的prompt，模型便不会输出了。当然，LLM幻觉prompt只能最大程度抑制，无法避免
    这时候，就需要人为写死程序避免LLM幻觉了
    """
    system_prompt = SystemMessage(content=(
        "你是一个知识问答助手。你只能根据以下【上下文】中的信息来回答问题。\n"
        "重要规则：\n"
        "1. 如果上下文中没有包含相关信息，你必须回答【抱歉，知识库中没有相关信息】\n"
        "2. 禁止使用你自己的知识来回答\n"
        "3. 禁止推测或补充上下文中没有的内容\n"
        f"\n【上下文】：\n{state['context']}\n"
        "【重要提醒】：只使用上下文中的信息，不要使用任何外部知识。"
    ))

    # 上下文为空，直接不走LLM，降级为固定回答
    if not state["context"]:
        return {"messages": ["抱歉，知识库中没有相关信息"]}
    else:
        response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# ==================== 6. 运行 ====================

if __name__ == "__main__":
    import sys

    # 支持命令行参数：--add "新文档内容" 直接添加文档
    if len(sys.argv) >= 3 and sys.argv[1] == "--add":
        add_document(sys.argv[2])
        # 添加后重建知识库
        build_knowledge_base()
        sys.exit(0)

    # 构建知识库（如果 documents.json 有变动，会重新生成）
    build_knowledge_base()

    while True:
        question = input("\n请输入问题（输入 quit 退出）：").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        print(f"\n{'=' * 60}")
        print(f"❓ 问题：{question}")
        print(f"{'=' * 60}")

        last_s = None
        for s in app.stream({"messages": [HumanMessage(content=question)]}, stream_mode="values"):
            if "context" in s and s["context"]:
                print(f"\n📄 [检索到的上下文]：\n{s['context'][:200]}...")
            last_s = s

        answer = last_s["messages"][-1]
        print(f"\n💬 回答：")
        answer.pretty_print()
