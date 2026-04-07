# Project 3 — Tool-Using Agent

ReAct agent that uses tools to answer questions requiring
current information, calculations, or domain knowledge.

## Tools
- search_web — DuckDuckGo web search
- calculate — safe math expression evaluator
- get_current_datetime — current date/time
- analyze_text — text statistics
- lookup_knowledge_base — distributed systems knowledge
- estimate_training_time — ML training time estimator

## Agent Behavior
- Selects correct tool for each question automatically
- Makes multiple tool calls when needed
- Falls back to own knowledge when tools return no results
- Maintains multi-turn conversation memory

## Results
- 7/7 test questions answered correctly
- Multi-turn memory: remembered training split across questions
- Graceful fallback: answered LangGraph question from knowledge when search failed

## How to Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-anthropic langchain-community \
    langgraph python-dotenv requests beautifulsoup4
echo "ANTHROPIC_API_KEY=your_key" > .env
python agent.py
```

## Tech Stack
- LangGraph — ReAct agent framework
- Claude Sonnet — reasoning and tool selection
- Custom tools — 6 domain-specific tools
