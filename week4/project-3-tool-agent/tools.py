import requests
import json
import math
from datetime import datetime
from langchain_core.tools import tool

# ── Tool 1 — Web Search ──────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic.
    Use this when you need up-to-date information or facts you don't know."""
    try:
        # Using DuckDuckGo instant answer API — free, no key needed
        url    = "https://api.duckduckgo.com/"
        params = {
            "q":      query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        response = requests.get(url, params=params, timeout=10)
        data     = response.json()

        results = []

        # Abstract (main answer)
        if data.get("Abstract"):
            results.append(f"Summary: {data['Abstract']}")

        # Related topics
        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text']}")

        # Answer (for simple factual queries)
        if data.get("Answer"):
            results.append(f"Answer: {data['Answer']}")

        if results:
            return "\n".join(results)
        else:
            return f"No direct results found for '{query}'. Try a more specific query."

    except Exception as e:
        return f"Search failed: {str(e)}"

# ── Tool 2 — Calculator ──────────────────────────────────
@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations.
    Use this for any math operations — arithmetic, percentages, etc.
    Input should be a valid Python math expression like '2 + 2' or 'math.sqrt(16)'"""
    try:
        # Safe evaluation — only allow math operations
        allowed = {
            "__builtins__": {},
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
        }
        result = eval(expression, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

# ── Tool 3 — Get Current Date/Time ───────────────────────
@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time.
    Use this when the user asks about the current date, time, or year."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} (local time)"

# ── Tool 4 — Text Analysis ───────────────────────────────
@tool
def analyze_text(text: str) -> str:
    """Analyze text and return statistics.
    Use this when asked to analyze, count, or get statistics about text."""
    words      = text.split()
    sentences  = text.count('.') + text.count('!') + text.count('?')
    paragraphs = text.count('\n\n') + 1
    chars      = len(text)
    unique     = len(set(word.lower().strip('.,!?') for word in words))

    return f"""Text Analysis:
- Characters: {chars}
- Words: {len(words)}
- Unique words: {unique}
- Sentences: {sentences}
- Paragraphs: {paragraphs}
- Average word length: {sum(len(w) for w in words) / len(words):.1f} chars
- Vocabulary richness: {unique/len(words):.1%}"""

# ── Tool 5 — Knowledge Base Lookup ───────────────────────
@tool
def lookup_knowledge_base(topic: str) -> str:
    """Look up information about distributed systems and ML topics.
    Use this for questions about Kubernetes, Kafka, CAP theorem, Docker, MLOps, RAG."""

    knowledge = {
        "kubernetes": """Kubernetes is a container orchestration platform.
        Key concepts: Pods, Deployments, Services, Namespaces.
        Uses declarative model — describe desired state, K8s maintains it.
        Control plane: API server, etcd, scheduler, controller manager.""",

        "kafka": """Apache Kafka is a distributed event streaming platform.
        Core concepts: Topics, Partitions, Producers, Consumers, Consumer Groups.
        Partitions are ordered, immutable sequences of records.
        Pull-based model — consumers control read position via offsets.""",

        "cap theorem": """CAP theorem: distributed systems can only guarantee 2 of 3:
        Consistency, Availability, Partition Tolerance.
        Partition tolerance is mandatory → real choice is CP vs AP.
        CP examples: HBase, Zookeeper. AP examples: Cassandra, CouchDB.""",

        "docker": """Docker containerizes applications with all dependencies.
        Key: Images (templates), Containers (running instances), Dockerfile.
        Containers share host OS kernel — lightweight vs VMs.
        Layer caching makes builds efficient.""",

        "mlops": """MLOps applies DevOps to ML systems.
        Key practices: experiment tracking, model versioning, automated pipelines.
        Data drift: input distribution changes. Concept drift: input-output relationship changes.
        Tools: MLflow, Weights & Biases, Feast, Seldon.""",

        "rag": """RAG combines retrieval with LLM generation.
        Steps: load documents, chunk, embed, store in vector DB, retrieve, generate.
        Reduces hallucinations by grounding answers in retrieved facts.
        Vector stores: ChromaDB, Pinecone, Weaviate.""",
    }

    topic_lower = topic.lower()
    for key, value in knowledge.items():
        if key in topic_lower or topic_lower in key:
            return value

    return f"No information found for '{topic}' in knowledge base. Try: kubernetes, kafka, cap theorem, docker, mlops, rag"

# ── Tool 6 — ML Model Estimator ──────────────────────────
@tool
def estimate_training_time(
    num_samples: int,
    num_epochs: int,
    batch_size: int,
    ms_per_batch: float = 50.0
) -> str:
    """Estimate ML model training time given dataset and training parameters.
    Use this when asked about how long training will take."""

    batches_per_epoch = math.ceil(num_samples / batch_size)
    total_batches     = batches_per_epoch * num_epochs
    total_ms          = total_batches * ms_per_batch
    total_seconds     = total_ms / 1000
    total_minutes     = total_seconds / 60

    return f"""Training Time Estimate:
- Samples: {num_samples}
- Epochs: {num_epochs}
- Batch size: {batch_size}
- Batches per epoch: {batches_per_epoch}
- Total batches: {total_batches}
- Estimated time: {total_seconds:.1f} seconds ({total_minutes:.1f} minutes)
- Note: Actual time varies by hardware (CPU vs GPU) and model size"""

# Export all tools
TOOLS = [
    search_web,
    calculate,
    get_current_datetime,
    analyze_text,
    lookup_knowledge_base,
    estimate_training_time
]