import requests
import math
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
        if data.get("Answer"):
            results.append(f"Answer: {data['Answer']}")
        return "\n".join(results) if results else f"No results for '{query}'"
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def lookup_knowledge_base(topic: str) -> str:
    """Look up information about distributed systems, ML, and cloud-native topics."""
    knowledge = {
        "kubernetes": "Kubernetes orchestrates containers declaratively. Key: Pods, Deployments, Services, Namespaces. Auto-scaling, self-healing. CP system — prioritizes consistency.",
        "docker swarm": "Docker Swarm is Docker's native clustering tool. Simpler than Kubernetes, built into Docker Engine. Uses overlay networks. Less feature-rich but easier to operate for small clusters.",
        "kafka": "Apache Kafka is a distributed event streaming platform. Core: Topics, Partitions, Producers, Consumers. Pull-based with offsets. Best for real-time pipelines and event sourcing.",
        "cap theorem": "CAP theorem: distributed systems guarantee only 2 of: Consistency, Availability, Partition Tolerance. CP: HBase, Zookeeper. AP: Cassandra, CouchDB.",
        "docker": "Docker containerizes apps with all dependencies. Images, Containers, Dockerfile. Lightweight vs VMs — shares host kernel. Layer caching for fast rebuilds.",
        "mlops": "MLOps applies DevOps to ML. Key: experiment tracking, model versioning, automated pipelines. Tools: MLflow, W&B, Seldon. Main failure modes: data drift, concept drift.",
        "rag": "RAG combines retrieval with LLM generation. Steps: chunk, embed, store in vector DB, retrieve, generate. Reduces hallucinations by grounding answers in facts.",
        "langgraph": "LangGraph builds stateful multi-agent systems as graphs. Nodes are actions, edges are transitions. Key features: MemorySaver for persistence, interrupt() for human-in-the-loop, streaming.",
    }
    topic_lower = topic.lower()
    for key, value in knowledge.items():
        if key in topic_lower or topic_lower in key:
            return value
    return f"No information found for '{topic}'. Available: kubernetes, docker swarm, kafka, cap theorem, docker, mlops, rag, langgraph"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        allowed = {"__builtins__": {}, "math": math}
        result = eval(expression, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} (local time)"


TOOLS = [search_web, lookup_knowledge_base, calculate, get_current_datetime]
