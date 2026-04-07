import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from documents import DOCUMENTS

load_dotenv()

# ── Initialize LLM ───────────────────────────────────────
llm = ChatAnthropic(
    model      = "claude-sonnet-4-20250514",
    api_key    = os.getenv("ANTHROPIC_API_KEY"),
    max_tokens = 1024,
    temperature = 0.0
)

# ── Initialize Embeddings ────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)
print("Embedding model loaded")

# ── Step 1 — Load and Chunk Documents ───────────────────
def load_documents():
    print("\nLoading documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 50
    )

    all_chunks = []
    for doc in DOCUMENTS:
        chunks = splitter.create_documents(
            texts    = [doc["content"]],
            metadatas= [{"id": doc["id"], "title": doc["title"]}]
        )
        all_chunks.extend(chunks)
        print(f"  {doc['title']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks

# ── Step 2 — Build Vector Store ──────────────────────────
def build_vector_store(chunks):
    print("\nBuilding vector store...")

    # Clear existing db to avoid duplicates
    import shutil
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    vector_store = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        persist_directory = "./chroma_db"
    )
    print(f"Vector store built with {vector_store._collection.count()} embeddings")
    return vector_store

# ── Step 3 — Retrieval ───────────────────────────────────
def retrieve(vector_store, query, k=3):
    results = vector_store.similarity_search_with_score(query, k=k)
    return results

# ── Step 4 — Generation ──────────────────────────────────
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions about 
distributed systems and MLOps. Answer ONLY based on the provided context. 
If the context doesn't contain enough information to answer, say so clearly.
Do not make up information."""),
    ("human", """Context:
{context}

Question: {question}

Answer based on the context above:""")
])

rag_chain = rag_prompt | llm | StrOutputParser()

def answer_question(vector_store, question):
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")

    # Retrieve with score threshold
    results = retrieve(vector_store, question, k=3)

    # Filter out low relevance results (high score = less similar in chromadb)
    SCORE_THRESHOLD = 2.0
    filtered = [(doc, score) for doc, score in results if score < SCORE_THRESHOLD]

    if not filtered:
        print("\n⚠️  No relevant context found in documents")
        print("Answer: I don't have information about this topic in my knowledge base.")
        return None

    print(f"\n📚 Retrieved {len(filtered)} relevant chunks (filtered from {len(results)}):")
    for doc, score in filtered:
        print(f"  [{score:.3f}] {doc.metadata['title']} — {doc.page_content[:80]}...")

    context = "\n\n".join([doc.page_content for doc, _ in filtered])

    print(f"\n💬 Answer:")
    answer = rag_chain.invoke({
        "context":  context,
        "question": question
    })
    print(answer)
    return answer

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Build knowledge base
    chunks       = load_documents()
    vector_store = build_vector_store(chunks)

    # Test questions
    questions = [
        "What is the CAP theorem and what are its trade-offs?",
        "How does Kafka handle message ordering?",
        "What is the difference between a service mesh and an API gateway?",
        "How does Raft prevent split-brain scenarios?",
        "What causes model drift in production ML systems?",
        "How does RAG reduce hallucinations?",
        "What are the core steps in badminton?"
    ]

    for question in questions:
        answer_question(vector_store, question)
        print()