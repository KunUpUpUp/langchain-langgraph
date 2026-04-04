"""
LangGraph 示例 - 智能客服路由系统

核心概念演示：
  StateGraph   → 定义状态图（节点 + 边）
  State        → 类型化的共享状态（消息列表 + 分类结果 + 路由目标）
  Node         → 图中的处理节点（每个节点是一个函数）
  Edge         → 节点间的转移（普通边 / 条件边）
  ToolNode     → LangGraph 内置的工具调用节点

与 LangChain Agent 的关键区别：
  LangChain Agent = 单个 LLM 循环（隐式状态机）
  LangGraph       = 你自己画的流程图（显式状态机，可以多分支、多 agent 协作）
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Literal, TypedDict
import json
import re


# ========== 1. 定义状态 ==========
class ServiceState(TypedDict):
    """
    图中所有节点共享的状态。
    必须用 TypedDict，LangGraph 才能识别 Annotated reducer。
    用 dict 的话 add_messages 注解不会生效，messages 会被覆盖而非追加。
    """
    messages: Annotated[list, add_messages]
    category: str          # 意图分类结果: tech / billing / general
    confidence: float      # 分类置信度
    final_response: str    # 最终回复


# ========== 2. 工具定义 ==========
@tool
def query_order_status(order_id: str) -> str:
    """查询订单状态，需要订单号"""
    orders = {
        "ORD-001": "已发货，预计明天到达",
        "ORD-002": "正在打包中，预计今天发出",
        "ORD-003": "已签收",
    }
    return orders.get(order_id, f"订单 {order_id} 未找到")


@tool
def check_refund_policy(product_type: str) -> str:
    """查询退款政策"""
    policies = {
        "电子产品": "7天无理由退货，15天质量问题包换",
        "服装": "30天无理由退换，需保留吊牌",
        "食品": "不支持无理由退货，质量问题48小时内受理",
    }
    return policies.get(product_type, f"{product_type} 的退款政策暂未录入")


@tool
def get_faq(question: str) -> str:
    """查询常见问题解答"""
    faqs = {
        "配送": "默认顺丰包邮，偏远地区可能需要额外运费，一般3-5个工作日送达。",
        "会员": "注册即享9折优惠，消费满500元升级为银牌会员享8.5折。",
        "发票": "下单时可申请电子发票，纸质发票请联系客服单独申请。",
    }
    for key, value in faqs.items():
        if key in question:
            return value
    return "暂未收录该问题，建议联系人工客服。"


tools = [query_order_status, check_refund_policy, get_faq]


# ========== 3. LLM ==========
llm = ChatOpenAI(model="qwen-plus", temperature=0).bind_tools(tools)

def agent_node(state: ServiceState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ========== 4. 节点定义 ==========
def classify_intent(state: ServiceState) -> dict:
    """
    节点1: 意图分类
    用一个独立的 LLM 调用判断用户问题属于哪个类别。
    这是 LangGraph 的优势——可以插入任意的中间处理节点。
    """
    last_msg = state["messages"][-1].content

    classifier = ChatOpenAI(model="qwen-plus", temperature=0)
    result = classifier.invoke([
        SystemMessage(content="""将用户问题分类为以下类别之一，只回复 JSON：
- tech: 技术问题、产品使用、故障排查
- billing: 订单查询、退款、价格、支付
- general: 配送、会员、发票等一般咨询"""),
        HumanMessage(content=last_msg),
    ])

    try:
        # LLM 可能返回 ```json ... ``` 包裹的内容，先清理
        text = re.sub(r"```(?:json)?\s*", "", result.content).strip().rstrip("`")
        data = json.loads(text)
        category = data.get("category", "general")
        confidence = float(data.get("confidence", 0.8))
    except (json.JSONDecodeError, AttributeError, ValueError):
        category = "general"
        confidence = 0.5

    return {"category": category, "confidence": confidence}


def route_by_category(state: ServiceState) -> Literal["tech_handler", "billing_handler", "general_handler"]:
    """
    条件边函数: 根据分类结果决定走哪个分支。
    返回值必须是一个节点名称字符串，LangGraph 会自动跳转到对应节点。
    """
    valid = {"tech", "billing", "general"}
    category = state.get("category", "general")
    if category not in valid:
        category = "general"
    return f"{category}_handler"


def tech_handler(state: ServiceState) -> dict:
    """节点: 技术支持处理"""
    messages = state["messages"] + [
        SystemMessage(content="你是技术支持专家。如果需要查资料请使用工具，否则直接回答。")
    ]
    return {"messages": messages}


def billing_handler(state: ServiceState) -> dict:
    """节点: 订单/售后处理"""
    messages = state["messages"] + [
        SystemMessage(content="你是订单和售后专员。使用工具查询订单或退款政策，然后给出完整回复。")
    ]
    return {"messages": messages}


def general_handler(state: ServiceState) -> dict:
    """节点: 一般咨询处理"""
    messages = state["messages"] + [
        SystemMessage(content="你是通用客服。使用工具查询FAQ或订单信息，礼貌地回答用户问题。")
    ]
    return {"messages": messages}


# ========== 5. 构建图 ==========
def build_graph():
    """
    构建完整的状态图：

                    START
                      |
                      v
                classify_intent    ← 节点: 意图分类
                      |
                      v
                route_by_category  ← 条件边: 三路分支
               /        |         \
              v         v          v
    tech_handler  billing_handler  general_handler  ← 节点: 专用处理
         \          |          /
          v         v         v
            tools_condition    ← 条件边: 是否需要调用工具
           /             \
          v               v
     tool_node        respond   ← 节点: 工具调用 / 生成回复
          |               |
          v               v
          →→→ respond ←←←     ← 回到 LLM 看是否还要继续调工具
                    |
                    v
                   END
    """
    graph = StateGraph(ServiceState)

    # 添加节点
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("tech_handler", tech_handler)
    graph.add_node("billing_handler", billing_handler)
    graph.add_node("general_handler", general_handler)
    graph.add_node("agent", agent_node)             # LLM 节点（绑定了工具）
    graph.add_node("tools", ToolNode(tools))  # 工具执行节点（LangGraph 内置）
    graph.add_node("respond", lambda state: {
        "final_response": state["messages"][-1].content
    })

    # 添加边
    graph.add_edge(START, "classify_intent")

    # 条件边: 分类 → 分支
    graph.add_conditional_edges(
        "classify_intent",
        route_by_category,
        ["tech_handler", "billing_handler", "general_handler"],
    )

    # 三个分支 → agent（LLM）
    graph.add_edge("tech_handler", "agent")
    graph.add_edge("billing_handler", "agent")
    graph.add_edge("general_handler", "agent")

    # 条件边: agent → tools 或 respond
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "respond"})

    # tools → 回到 agent（让 LLM 看工具结果，决定是否继续）
    graph.add_edge("tools", "agent")

    graph.add_edge("respond", END)

    return graph.compile()


# ========== 6. 运行 ==========
def main():
    graph = build_graph()

    test_queries = [
        "我的订单 ORD-001 到哪了？",
        "这个退款政策是什么？",
        "你们配送一般多长时间？",
        "怎么重置密码？",
    ]

    print("=" * 60)
    print("LangGraph 智能客服路由系统")
    print("=" * 60)

    for query in test_queries:
        print(f"\n用户: {query}")

        result = graph.invoke({
            "messages": [HumanMessage(content=query)],
            "category": "",
            "confidence": 0.0,
            "final_response": "",
        })

        category = result.get("category", "unknown")
        confidence = result.get("confidence", 0.0)
        response = result.get("final_response", "")

        print(f"  [分类: {category}, 置信度: {confidence:.2f}]")
        print(f"助手: {response}")
        print("-" * 60)


if __name__ == "__main__":
    main()
