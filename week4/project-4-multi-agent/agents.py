import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from tools import RESEARCH_TOOLS, WRITER_TOOLS

load_dotenv()

# ── Initialize Claude ────────────────────────────────────
def get_llm():
    return ChatAnthropic(
        model       = "claude-sonnet-4-20250514",
        api_key     = os.getenv("ANTHROPIC_API_KEY"),
        max_tokens  = 4096,
        temperature = 0.0
    )

# ── Research Agent ───────────────────────────────────────
RESEARCH_SYSTEM_PROMPT = """You are an expert Research Agent specializing in 
distributed systems, MLOps, and AI technologies.

Your job is to:
1. Research the given topic thoroughly using available tools
2. Look up relevant information from the knowledge base
3. Search the web for additional context if needed
4. Compile structured research notes with key findings

Always structure your research output as:
- Key Facts: bullet points of important facts
- Strengths: what makes this technology good
- Weaknesses: limitations or challenges
- Use Cases: when to use this technology
- Comparison Points: how it differs from alternatives

Be thorough and factual. Cite what you found from tools."""

def run_research_agent(topic: str) -> str:
    """Run the research agent on a given topic"""
    print(f"\n🔍 Research Agent starting on: {topic}")

    llm            = get_llm()
    research_agent = create_react_agent(
        model  = llm,
        tools  = RESEARCH_TOOLS,
        prompt = RESEARCH_SYSTEM_PROMPT
    )

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=f"Research this topic thoroughly: {topic}")
    ]

    result = research_agent.invoke({"messages": messages})

    # Show tool calls
    print("\n📚 Research Agent tool calls:")
    for msg in result["messages"]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  🔧 {tc['name']}({tc['args']})")

    # Extract final research notes
    final = result["messages"][-1].content
    print(f"\n✅ Research complete ({len(final)} chars)")
    return final

# ── Writer Agent ─────────────────────────────────────────
WRITER_SYSTEM_PROMPT = """You are a Technical Writer Agent.

CRITICAL INSTRUCTIONS:
- You MUST call the format_report tool with the actual report content
- Do NOT describe what you will write — actually write it
- Do NOT summarize the report — produce the full report text
- The format_report tool parameters must contain the FULL content, not descriptions

When writing the report:
- summary parameter: Write 2-3 actual sentences summarizing the topic
- sections parameter: Write the FULL analysis with headers and content
- conclusion parameter: Write 2-3 actual sentences with recommendations
- title parameter: The report title

Example of WRONG behavior:
  format_report(summary="This section covers the key points...")
  
Example of CORRECT behavior:
  format_report(summary="Kubernetes is a container orchestration platform 
  that automates deployment and scaling of containerized applications. 
  It has become the industry standard for managing microservices at scale.")

Always produce the actual written content, not descriptions of content."""

def run_writer_agent(topic: str, research_notes: str) -> str:
    """Run the writer agent to create a report from research notes"""
    print(f"\n✍️  Writer Agent starting on: {topic}")

    llm = get_llm()

    # Direct LLM call — no tools needed for writing
    from langchain_core.messages import SystemMessage, HumanMessage
    
    messages = [
        SystemMessage(content="""You are a Technical Writer. 
Write professional, detailed reports in Markdown format.
Always write the FULL report content — never describe or summarize what you will write."""),
        HumanMessage(content=f"""Write a complete professional report on: {topic}

Use this research:
{research_notes}

Format your response as a complete Markdown report with:
# {topic}

## Executive Summary
[2-3 sentences]

## Architecture & Core Concepts
[detailed content]

## Strengths
[detailed content]

## Weaknesses  
[detailed content]

## Use Cases
[detailed content]

## Conclusion
[2-3 sentences with recommendations]

Write the FULL content now:""")
    ]

    response = llm.invoke(messages)
    report   = response.content

    print(f"\n✅ Writing complete ({len(report)} chars)")
    return report