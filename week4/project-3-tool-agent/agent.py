import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from tools import TOOLS

load_dotenv()

# ── Initialize Claude ────────────────────────────────────
llm = ChatAnthropic(
    model      = "claude-sonnet-4-20250514",
    api_key    = os.getenv("ANTHROPIC_API_KEY"),
    max_tokens = 4096,
    temperature = 0.0
)

# ── System prompt ────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful AI assistant with access to several tools.

Your available tools:
- search_web: Search for current information on the web
- calculate: Perform mathematical calculations
- get_current_datetime: Get current date and time
- analyze_text: Analyze text statistics
- lookup_knowledge_base: Look up distributed systems and ML topics

Guidelines:
- Always use tools when you need current information or calculations
- Use lookup_knowledge_base for questions about Kubernetes, Kafka, Docker, MLOps, RAG
- Use search_web for topics not in the knowledge base
- Think step by step before answering
- Be concise but complete in your final answers
"""

# ── Create Agent using LangGraph ─────────────────────────
agent = create_react_agent(
    model  = llm,
    tools  = TOOLS,
    prompt = SYSTEM_PROMPT
)

# ── Chat History ─────────────────────────────────────────
chat_history = []

def run_agent(question: str):
    print(f"\n{'='*60}")
    print(f"USER: {question}")
    print(f"{'='*60}")

    # Build messages
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=question))

    # Run agent
    result = agent.invoke({"messages": messages})

    # Extract final answer
    final_message = result["messages"][-1]
    answer = final_message.content

    # Show agent reasoning
    print("\n🤔 Agent reasoning:")
    for msg in result["messages"]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  🔧 Tool call: {tc['name']}({tc['args']})")
        elif msg.__class__.__name__ == 'ToolMessage':
            print(f"  📥 Tool result: {str(msg.content)[:150]}...")

    # Update chat history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(final_message)

    print(f"\n💬 FINAL ANSWER: {answer}")
    return answer

# ── Test Questions ───────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What is the CAP theorem and give me examples of CP and AP systems?",
        "If I have 3000 training samples and use 80% for training, how many samples are in each split?",
        "What is today's date?",
        "How many words are in this text: 'Kubernetes orchestrates containerized applications across clusters using a declarative model'?",
        "What is LangGraph used for?",
        "Based on the training split you calculated earlier, if each sample takes 2ms to process, how long will training take in seconds?",
        "How long will it take to train a model on 10000 samples for 3 epochs with batch size 32?",
    ]

    for question in questions:
        run_agent(question)
        print()