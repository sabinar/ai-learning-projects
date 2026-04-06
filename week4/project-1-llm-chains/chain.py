import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env
load_dotenv()

# Initialize Claude
#llm = ChatAnthropic(
#    model="claude-sonnet-4-20250514",
#    api_key=os.getenv("ANTHROPIC_API_KEY"),
#    max_tokens=1024
#)

# Two versions — creative vs deterministic
llm_creative     = ChatAnthropic(
    model        = "claude-sonnet-4-20250514",
    api_key      = os.getenv("ANTHROPIC_API_KEY"),
    max_tokens   = 1024,
    temperature  = 0.9   # more creative, varied output
)

llm_deterministic = ChatAnthropic(
    model        = "claude-sonnet-4-20250514",
    api_key      = os.getenv("ANTHROPIC_API_KEY"),
    max_tokens   = 1024,
    temperature  = 0.0   # consistent, reproducible output
)

# Use deterministic for production chains
llm = llm_deterministic

# ── Chain 1 — Summarize ──────────────────────────────────
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at summarizing technical content concisely."),
    ("human", "Summarize the following text in 3-4 sentences:\n\n{text}")
])

summarize_chain = summarize_prompt | llm | StrOutputParser()

# ── Chain 2 — Extract Topics ─────────────────────────────
topics_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at identifying key technical topics."),
    ("human", "Extract 3-5 key topics from this summary as a bullet list:\n\n{summary}")
])

topics_chain = topics_prompt | llm | StrOutputParser()

# ── Chain 3 — Generate Questions ─────────────────────────
questions_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at generating insightful questions for deeper learning."),
    ("human", "Generate 3 thought-provoking questions based on these topics:\n\n{topics}")
])

questions_chain = questions_prompt | llm | StrOutputParser()

# ── Full Pipeline ─────────────────────────────────────────
def run_pipeline(text):
    print("\n" + "="*60)
    print("INPUT TEXT")
    print("="*60)
    print(text[:200] + "..." if len(text) > 200 else text)

    # Step 1 — Summarize
    print("\n" + "="*60)
    print("STEP 1 — SUMMARY")
    print("="*60)
    summary = summarize_chain.invoke({"text": text})
    print(summary)

    # Step 2 — Extract topics
    print("\n" + "="*60)
    print("STEP 2 — KEY TOPICS")
    print("="*60)
    topics = topics_chain.invoke({"summary": summary})
    print(topics)

    # Step 3 — Generate questions
    print("\n" + "="*60)
    print("STEP 3 — QUESTIONS FOR DEEPER LEARNING")
    print("="*60)
    questions = questions_chain.invoke({"topics": topics})
    print(questions)

    return {
        "summary": summary,
        "topics": topics,
        "questions": questions
    }


def run_pipeline_streaming(text):
    """Same pipeline but streams output in real time"""
    print("\n" + "="*60)
    print("STREAMING PIPELINE")
    print("="*60)

    # Step 1 — Stream summary
    print("\n📝 Generating summary...")
    summary_chunks = []
    for chunk in summarize_chain.stream({"text": text}):
        print(chunk, end="", flush=True)
        summary_chunks.append(chunk)
    summary = "".join(summary_chunks)

    # Step 2 — Stream topics
    print("\n\n🏷️  Extracting topics...")
    topics_chunks = []
    for chunk in topics_chain.stream({"summary": summary}):
        print(chunk, end="", flush=True)
        topics_chunks.append(chunk)
    topics = "".join(topics_chunks)

    # Step 3 — Stream questions
    print("\n\n❓ Generating questions...")
    for chunk in questions_chain.stream({"topics": topics}):
        print(chunk, end="", flush=True)

    print("\n\n✅ Done")

def compare_temperatures(text):
    """Show how temperature affects output"""
    print("\n" + "="*60)
    print("TEMPERATURE COMPARISON")
    print("="*60)

    summarize_creative = (
        ChatPromptTemplate.from_messages([
            ("system", "You are an expert at summarizing technical content."),
            ("human", "Summarize in 2 sentences:\n\n{text}")
        ]) | llm_creative | StrOutputParser()
    )

    summarize_deterministic = (
        ChatPromptTemplate.from_messages([
            ("system", "You are an expert at summarizing technical content."),
            ("human", "Summarize in 2 sentences:\n\n{text}")
        ]) | llm_deterministic | StrOutputParser()
    )

    print("\n🌡️  Temperature 0.9 (creative):")
    print(summarize_creative.invoke({"text": text}))

    print("\n🌡️  Temperature 0.0 (deterministic):")
    print(summarize_deterministic.invoke({"text": text}))

    print("\n🌡️  Temperature 0.0 again (should be identical):")
    print(summarize_deterministic.invoke({"text": text}))    

# ── Test with distributed systems text ───────────────────
if __name__ == "__main__":
    test_text = """
    Distributed systems are collections of independent computers that appear 
    to users as a single coherent system. The CAP theorem states that a 
    distributed system can only guarantee two of three properties: consistency,
    availability, and partition tolerance. Modern distributed systems like 
    Kubernetes orchestrate containerized workloads across clusters, handling 
    failures automatically through health checks and self-healing mechanisms.
    Service meshes like Istio manage communication between microservices,
    providing observability, traffic management, and security policies.
    Event-driven architectures using message brokers like Apache Kafka enable
    loose coupling between services, allowing them to scale independently
    and handle failures gracefully through replay and dead letter queues.
    """

    #result = run_pipeline(test_text)


# Add this at the bottom of the file
if __name__ == "__main__":
    test_text = """
    Distributed systems are collections of independent computers that appear 
    to users as a single coherent system. The CAP theorem states that a 
    distributed system can only guarantee two of three properties: consistency,
    availability, and partition tolerance. Modern distributed systems like 
    Kubernetes orchestrate containerized workloads across clusters, handling 
    failures automatically through health checks and self-healing mechanisms.
    Service meshes like Istio manage communication between microservices,
    providing observability, traffic management, and security policies.
    Event-driven architectures using message brokers like Apache Kafka enable
    loose coupling between services, allowing them to scale independently
    and handle failures gracefully through replay and dead letter queues.
    """

    # Run normal pipeline
    result = run_pipeline(test_text)

    # Run streaming pipeline
    print("\n\n" + "="*60)
    print("NOW WITH STREAMING")
    print("="*60)
    run_pipeline_streaming(test_text)    

    compare_temperatures(test_text)

    