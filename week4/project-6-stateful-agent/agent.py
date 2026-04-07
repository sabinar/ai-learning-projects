import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are a helpful research assistant with a persistent memory of our conversation.

You have access to tools to look up information. Use them when needed.

Important behaviors:
- Remember everything discussed earlier in our conversation
- When asked to compare topics, refer back to what you already found
- When asked to summarize, cover everything from the entire conversation
- Be concise but thorough
"""

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4096,
    temperature=0.0,
)

# MemorySaver stores the full graph state (all messages + tool calls)
# keyed by thread_id — no manual history management needed
checkpointer = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)


def run_agent(question: str, thread_id: str = "default") -> str:
    # thread_id is the key — same thread_id = same conversation memory
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    return result["messages"][-1].content


def get_history(thread_id: str = "default") -> list[dict]:
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    if not state.values:
        return []

    history = []
    for msg in state.values.get("messages", []):
        role = msg.__class__.__name__.replace("Message", "").lower()
        if role in ("human", "ai"):
            history.append({
                "role": "user" if role == "human" else "assistant",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            })
    return history
