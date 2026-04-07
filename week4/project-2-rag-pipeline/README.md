# Project 2 — RAG Pipeline

Q&A system over custom documents using RAG.
Retrieves relevant context before answering — grounded in facts, not hallucinations.

## Architecture
Documents → Chunk → Embed → ChromaDB
↓
User question → Embed → Similarity search → Top chunks
↓
Claude answers using retrieved context

## Knowledge Base
8 documents covering: CAP Theorem, Kubernetes, Kafka, Service Mesh,
Raft Consensus, Docker, MLOps, RAG

## Results
- 6/6 questions answered correctly
- Correctly admitted knowledge gap for API gateway question
- Score threshold filtering removes irrelevant chunks

## How to Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-anthropic langchain-community \
    langchain-text-splitters chromadb sentence-transformers python-dotenv
echo "ANTHROPIC_API_KEY=your_key" > .env
python rag.py
```

## Tech Stack
- ChromaDB — vector store
- all-MiniLM-L6-v2 — embedding model
- Claude Sonnet — answer generation
- LangChain — pipeline orchestration
