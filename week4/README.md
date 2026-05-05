# Week 4 — LLM Apps & Agents

Building production-ready LLM applications: chains, RAG, tool-using agents, multi-agent systems, and stateful/human-in-the-loop workflows.

**Tech:** LangChain · LangGraph · Claude Sonnet (Anthropic API) · ChromaDB · FastAPI · Docker

---

## Projects

### [project-1-llm-chains](project-1-llm-chains/)
Multi-step reasoning pipeline using LangChain prompt chaining with Claude Sonnet. Three sequential LLM calls: summarize → extract topics → generate questions.

### [project-2-rag-pipeline](project-2-rag-pipeline/)
Retrieval-Augmented Generation (RAG) Q&A over custom documents. Embeds docs into ChromaDB, retrieves relevant chunks via similarity search, generates answers with Claude.

### [project-3-tool-agent](project-3-tool-agent/)
ReAct agent with 6 domain-specific tools (web search, calculator, datetime, text analysis, knowledge base, training time estimator). Uses LangGraph + Claude Sonnet for the agent loop.

### [project-4-multi-agent](project-4-multi-agent/)
Two specialized Claude Sonnet agents orchestrated in sequence: Research Agent gathers information, Writer Agent produces a polished report. Handles single topics and comparisons.

### [project-5-agent-api](project-5-agent-api/)
FastAPI REST wrapper around a Claude Sonnet ReAct agent with session-based conversation history. Dockerized for deployment.
- Endpoints: `POST /agent/ask`, `POST /agent/research`, `GET /agent/history`

### [project-6-stateful-agent](project-6-stateful-agent/)
Stateful Claude Sonnet agent using LangGraph's MemorySaver checkpointer. Memory persists across turns within a session — the agent remembers what was discussed earlier in the conversation.

### [project-7-hitl-agent](project-7-hitl-agent/)
Human-in-the-Loop Claude Sonnet agent with approval gates for sensitive operations (e.g. writing files). Uses LangGraph `interrupt_before` to pause execution and request user confirmation before irreversible actions.
