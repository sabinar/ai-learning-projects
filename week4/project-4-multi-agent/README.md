# Project 4 — Multi-Agent System

Two specialized agents orchestrated to research and write reports:
- Research Agent: gathers information using tools
- Writer Agent: transforms research into polished reports

## Pipelines

### Single Topic Pipeline
Research Agent runs 3 times (concepts, strengths/weaknesses, use cases)
→ Writer Agent produces comprehensive report

### Comparison Pipeline  
Research Agent runs on each technology separately
→ Writer Agent produces side-by-side comparison

## Results
- Kubernetes report: 10,509 chars from 11,981 chars of research
- K8s vs Docker Swarm: 11,438 chars comparison report
- Pipeline 1 time: ~117 seconds
- Pipeline 2 time: ~90 seconds

## How to Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-anthropic langgraph python-dotenv requests
echo "ANTHROPIC_API_KEY=your_key" > .env
python orchestrator.py
```

## Tech Stack
- LangGraph — agent framework
- Claude Sonnet — research and writing
- Custom tools — web search, knowledge base, comparison
