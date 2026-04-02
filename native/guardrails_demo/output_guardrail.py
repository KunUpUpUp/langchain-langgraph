"""输出护栏 - 在 LLM 输出返回给用户之前进行安全检查"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import re
import json


# ========== 1. PII（个人隐私信息）脱敏护栏 ==========
def pii_mask(output: str) -> str:
    """检测并遮蔽输出中的个人隐私信息"""
    # 手机号
    output = re.sub(r'1[3-9]\d{9}', '[手机号已隐藏]', output)
    # 邮箱
    output = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[邮箱已隐藏]', output)
    # 身份证号
    output = re.sub(r'\d{17}[\dXx]', '[身份证已隐藏]', output)
    # IP 地址
    output = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP已隐藏]', output)
    return output


# ========== 2. 基于 LLM 的输出安全审核 ==========
def llm_output_review(output: str) -> tuple[bool, str]:
    """使用 LLM 审核输出内容是否合规"""
    llm = ChatOpenAI(model="qwen-plus")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个输出内容安全审核员。检查 AI 的回复是否存在以下问题：
- 泄露敏感信息（密码、密钥、内部系统信息）
- 包含有害、歧视、暴力内容
- 产生幻觉（声称自己有情感/意识）
- 试图引导用户做危险操作

只回复 JSON：{{"safe": true/false, "reason": "原因", "suggestion": "修改建议"}}"""),
        ("user", "请审核以下 AI 回复：\n{output}"),
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"output": output})

    try:
        parsed = json.loads(result)
        return parsed["safe"], parsed.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        return True, ""


# ========== 3. 构建带输出护栏的完整链 ==========
def build_guarded_chain():
    """构建带输出护栏的 LangChain 链"""
    llm = ChatOpenAI(model="qwen-plus")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的助手。请详细回答用户的问题。"),
        ("user", "{input}"),
    ])

    # 输出护栏处理函数
    def output_guard(llm_output: str) -> str:
        # 第一层：PII 脱敏
        masked = pii_mask(llm_output)
        if masked != llm_output:
            print("[护栏] 已脱敏 PII 信息")

        # 第二层：LLM 审核
        safe, reason = llm_output_review(masked)
        if not safe:
            print(f"[护栏] 输出被拦截：{reason}")
            return "抱歉，该回复内容不符合安全规范，已被过滤。"

        return masked

    chain = prompt | llm | StrOutputParser() | RunnableLambda(output_guard)
    return chain


# ========== 4. 演示 ==========
if __name__ == "__main__":
    chain = build_guarded_chain()

    # 测试 PII 脱敏
    print("=== 测试 PII 脱敏 ===")
    response = chain.invoke({"input": "帮我生成一个示例用户信息，包含姓名、手机号13812345678、邮箱test@example.com"})
    print(f"最终输出: {response}\n")

    # 测试正常对话
    print("=== 测试正常对话 ===")
    response = chain.invoke({"input": "用 Python 写一个快速排序"})
    print(f"最终输出: {response}")
