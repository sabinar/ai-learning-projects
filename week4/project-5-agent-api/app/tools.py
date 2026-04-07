import requests
import math
from datetime import datetime
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic."""
    try:
        url    = "https://api.duckduckgo.com/"
        params = {
            "q":             query,
            "format":        "json",
            "no_html":       "1",
            "skip_disambig": "1"
        }
        response = requests.get(url, params=params, timeout=10)
        data     = response.json()
        results  = []
        if data.get("Abstract"):
            results.append(f"Summary: {data['Abstract']}")
        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text']}")
        if data.get("Answer"):
            results.append(f"Answer: {data['Answer']}")
        return "\n".join(results) if results else f"No results for '{query}'"
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def lookup_knowledge_base(topic: str) -> str:
    """Look up information about distributed systems and ML topics."""
    knowledge = {
        "kubernetes": "Kubernetes is a container orchestration platform. Key concepts: Pods, Deployments, Services. Uses declarative model. Strengths: Auto-scaling, self-healing. Best for: Large-scale production.",
        "kafka":      "Apache Kafka is a distributed event streaming platform. Core: Topics, Partitions, Producers, Consumers. Pull-based model with offsets. Best for: Real-time pipelines.",
        "cap theorem": "CAP theorem: distributed systems can only guarantee 2 of 3: Consistency, Availability, Partition Tolerance. CP examples: HBase, Zookeeper. AP examples: Cassandra, CouchDB.",
        "docker":     "Docker containerizes applications with all dependencies. Key: Images, Containers, Dockerfile. Lightweight vs VMs. Layer caching for efficient builds.",
        "mlops":      "MLOps applies DevOps to ML systems. Key: experiment tracking, model versioning, automated pipelines. Data drift and concept drift are main failure modes.",
        "rag":        "RAG combines retrieval with LLM generation. Steps: chunk, embed, store, retrieve, generate. Reduces hallucinations by grounding in facts.",
        "langgraph":  "LangGraph builds stateful multi-agent systems as graphs. Nodes are actions, edges are transitions. Supports cyclic workflows and human-in-the-loop.",
    }
    topic_lower = topic.lower()
    for key, value in knowledge.items():
        if key in topic_lower or topic_lower in key:
            return value
    return f"No information found for '{topic}'"

@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        allowed = {"__builtins__": {}, "math": math}
        result  = eval(expression, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} (local time)"

@tool
def analyze_text(text: str) -> str:
    """Analyze text and return statistics."""
    words     = text.split()
    chars     = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    unique    = len(set(w.lower().strip('.,!?') for w in words))
    return f"Characters: {chars}, Words: {len(words)}, Unique words: {unique}, Sentences: {sentences}"

TOOLS = [search_web, lookup_knowledge_base, calculate, get_current_datetime, analyze_text]