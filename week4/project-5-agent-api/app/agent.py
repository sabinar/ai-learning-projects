import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from app.tools import TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant with access to several tools.

Your available tools:
- search_web: Search for current information on the web
- calculate: Perform mathematical calculations
- get_current_datetime: Get current date and time
- analyze_text: Analyze text statistics
- lookup_knowledge_base: Look up distributed systems and ML topics

Guidelines:
- Always use tools when you need current information or calculations
- Use lookup_knowledge_base for questions about Kubernetes, Kafka, Docker, MLOps, RAG, LangGraph
- Use search_web for topics not in the knowledge base
- Think step by step before answering
- Be concise but complete in your final answers
"""

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4096,
    temperature=0.0,
)

agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
)

# Session history: { session_id: [HumanMessage, AIMessage, ...] }
_history: dict[str, list] = {}


def run_agent(question: str, session_id: str = "default") -> dict:
    history = _history.setdefault(session_id, [])

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(history)
    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})

    final_message = result["messages"][-1]
    answer = final_message.content

    # Track tool calls for the response metadata
    tool_calls_made = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc["name"])

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "tools_used": tool_calls_made,
        "session_id": session_id,
    }


def get_history(session_id: str = "default") -> list[dict]:
    history = _history.get(session_id, [])
    result = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


def clear_history(session_id: str = "default") -> None:
    _history.pop(session_id, None)
