import os
import requests
from datetime import datetime
from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        results = []
        if data.get("Abstract"):
            results.append(f"Summary: {data['Abstract']}")
        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text']}")
        return "\n".join(results) if results else f"No results for '{query}'"
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def lookup_knowledge_base(topic: str) -> str:
    """Look up information about distributed systems and ML topics."""
    knowledge = {
        "kafka": "Apache Kafka is a distributed event streaming platform. Core concepts: Topics, Partitions, Producers, Consumers, Consumer Groups. Pull-based model with offsets. Strengths: high throughput, fault tolerance, replay. Best for: real-time pipelines, event sourcing, log aggregation.",
        "kubernetes": "Kubernetes orchestrates containers declaratively. Key: Pods, Deployments, Services, Namespaces. Auto-scaling, self-healing. CP system. Best for: large-scale production workloads needing resilience.",
        "mlops": "MLOps applies DevOps to ML systems. Key practices: experiment tracking, model versioning, automated pipelines. Tools: MLflow, W&B, Seldon. Main failure modes: data drift, concept drift.",
        "rag": "RAG combines retrieval with LLM generation. Steps: chunk, embed, store in vector DB, retrieve, generate. Reduces hallucinations by grounding answers in retrieved facts. Tools: ChromaDB, Pinecone.",
        "langgraph": "LangGraph builds stateful multi-agent systems as graphs. Nodes are actions, edges are transitions. Key features: MemorySaver for persistence, interrupt() for human-in-the-loop, streaming support.",
    }
    topic_lower = topic.lower()
    for key, value in knowledge.items():
        if key in topic_lower or topic_lower in key:
            return value
    return f"No information found for '{topic}'. Available: kafka, kubernetes, mlops, rag, langgraph"


@tool
def save_report(filename: str, content: str) -> str:
    """Save a research report to a file on disk.
    WARNING: This is an irreversible action — requires human approval before executing."""
    os.makedirs("reports", exist_ok=True)
    filepath = os.path.join("reports", filename)
    with open(filepath, "w") as f:
        f.write(content)
        f.write(f"\n\n---\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return f"Report saved to {filepath}"


TOOLS = [search_web, lookup_knowledge_base, save_report]
