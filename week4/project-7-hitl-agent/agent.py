import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are a research assistant that can look up information and save reports to disk.

When asked to research a topic AND save a report:
1. First research the topic thoroughly using lookup_knowledge_base and search_web
2. Then call save_report with a clear filename and well-structured content

Always save reports in markdown format with clear headings.
"""

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4096,
    temperature=0.0,
)

checkpointer = MemorySaver()

# interrupt_before=["tools"] pauses the graph before the tools node fires.
# This gives us a chance to inspect pending tool calls and ask for approval
# before any tool — especially irreversible ones — actually executes.
agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    interrupt_before=["tools"],
)

SENSITIVE_TOOLS = {"save_report"}


def run_with_approval(question: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}

    # Initial invoke — runs until the first interrupt (before tools node)
    agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config,
    )

    # Loop: handle each interrupt until the graph finishes
    while True:
        state = agent.get_state(config)

        # No next nodes = graph is complete
        if not state.next:
            break

        # The last message is an AIMessage with pending tool_calls
        last_msg = state.values["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])

        # Split into sensitive vs safe tool calls
        sensitive = [tc for tc in tool_calls if tc["name"] in SENSITIVE_TOOLS]

        if sensitive:
            # Show the human what the agent wants to do
            tc = sensitive[0]
            print(f"\n{'='*55}")
            print("AGENT wants to perform an irreversible action:")
            print(f"  Tool     : {tc['name']}")
            print(f"  Filename : {tc['args'].get('filename', 'report.txt')}")
            print(f"\n  Content preview:")
            preview = tc["args"].get("content", "")[:500]
            print(f"{preview}...")
            print(f"{'='*55}")

            approval = input("\nApprove? (yes/no): ").strip().lower()

            if approval != "yes":
                # Inject fake ToolMessages so graph state stays consistent,
                # then return immediately with a clear rejection message.
                cancellations = [
                    ToolMessage(
                        content="Action cancelled by user.",
                        tool_call_id=tc["id"],
                    )
                    for tc in tool_calls
                ]
                agent.update_state(config, {"messages": cancellations})
                return f"Understood. The report was not saved. Research is complete but no file was written to disk."

        # Resume: if approved, tools execute normally.
        # If cancelled, fake ToolMessages are already in state — Claude will
        # see "Action cancelled by user" and respond accordingly.
        agent.invoke(None, config)

    all_messages = agent.get_state(config).values["messages"]
    # Walk backwards to find the last AIMessage that contains actual text.
    # Some AIMessages mix text + tool_use blocks — we extract just the text parts.
    from langchain_core.messages import AIMessage
    for msg in reversed(all_messages):
        if not isinstance(msg, AIMessage):
            continue
        if isinstance(msg.content, str) and msg.content.strip():
            return msg.content
        if isinstance(msg.content, list):
            texts = [
                block["text"]
                for block in msg.content
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip()
            ]
            if texts:
                return "\n".join(texts)
    return "No response generated."
