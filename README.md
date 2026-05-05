# AI Upskilling Projects

A structured 4-week AI/ML upskilling plan for a senior software engineer with a distributed systems background. Projects progress from ML fundamentals to production LLM agents.

---

## Week 2 — NLP with HuggingFace Transformers

Fine-tuning and applying pre-trained transformer models for core NLP tasks.

| Project | Description |
|---------|-------------|
| [project-1-nlp-pipeline](week2/project-1-nlp-pipeline/) | Five NLP tasks in one pipeline: sentiment, summarization, QA, zero-shot classification, text generation |
| [project-2-toxic-comment-classifier](week2/project-2-toxic-comment-classifier/) | Fine-tuned DistilBERT on Civil Comments dataset for toxic comment detection |
| [project-3-semantic-search](week2/project-3-semantic-search/) | Semantic similarity search using sentence embeddings |
| [project-4-tokenizer-explorer](week2/project-4-tokenizer-explorer/) | Deep dive into HuggingFace tokenization mechanics |
| [project-5-amazon-review-classifier](week2/project-5-amazon-review-classifier/) | Multi-class text classifier trained on Amazon reviews |

**Tech:** HuggingFace Transformers · DistilBERT · BART · GPT-2 · Trainer API

---

## Week 3 — MLOps: Serving & Pipelines

Deploying fine-tuned models as production APIs with containerization and automated ML pipelines.

| Project | Description |
|---------|-------------|
| [project-1-model-api](week3/project-1-model-api/) | FastAPI REST API serving DistilBERT sentiment analysis (~56ms/request) |
| [project-3-docker](week3/project-3-docker/) | Containerized sentiment API — 82% image size reduction with CPU-only PyTorch |
| [project-4-pipeline](week3/project-4-pipeline/) | Automated train → evaluate → deploy → monitor pipeline with drift detection |

**Tech:** FastAPI · Docker · DistilBERT · HuggingFace Trainer · Uvicorn

---

## Week 4 — LLM Apps & Agents

Building production-ready LLM applications: chains, RAG, tool-using agents, multi-agent systems, stateful memory, and human-in-the-loop workflows.

| Project | Description |
|---------|-------------|
| [project-1-llm-chains](week4/project-1-llm-chains/) | Multi-step reasoning with LangChain prompt chaining (summarize → topics → questions) |
| [project-2-rag-pipeline](week4/project-2-rag-pipeline/) | RAG Q&A over custom documents using ChromaDB + Claude |
| [project-3-tool-agent](week4/project-3-tool-agent/) | ReAct agent with 6 domain-specific tools via LangGraph |
| [project-4-multi-agent](week4/project-4-multi-agent/) | Research + Writer agents orchestrated in sequence |
| [project-5-agent-api](week4/project-5-agent-api/) | FastAPI + Docker wrapper around a tool-using agent with session history |
| [project-6-stateful-agent](week4/project-6-stateful-agent/) | Persistent multi-turn memory using LangGraph MemorySaver checkpointer |
| [project-7-hitl-agent](week4/project-7-hitl-agent/) | Human approval gates for irreversible actions using LangGraph interrupts |

**Tech:** LangChain · LangGraph · Claude Sonnet · ChromaDB · FastAPI · Docker

---

## Setup

Each project has its own virtual environment.

```bash
cd week4/project-7-hitl-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Anthropic API key to a `.env` file:

```
ANTHROPIC_API_KEY=your_key_here
```
