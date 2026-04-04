from typing import TypedDict, List, Sequence, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    """
    Annotated：给类型添加标签
    Sequence: 类似List，只不过可以塞更多类型
    BaseMessage: 消息基类，可以是HumanMessage、AIMessage、SystemMessage、ToolMessage
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """这是个加法算数工具"""
    return a + b

def llm(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="你是一个智能助手，可以在适当情况下调用工具")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def continue_node(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # last_message是一个对象，last_message.tool_calls是调用对象属性,if not last_message.tool_calls 等价于 if last_message.tool_calls is None
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

tools = [add]
model = ChatOpenAI(model="qwen-plus").bind_tools(tools)
graph = StateGraph(AgentState)
graph.add_node("llm", llm)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_edge("tools", "llm")
graph.add_conditional_edges(
    # start node
    "llm",
    continue_node,
    {
        "continue": "tools",
        "end": END
    }
)

app = graph.compile()
print_stream(app.stream({"messages": "帮我计算1+1,然后用结果再加上9算新结果"}, stream_mode="values"))