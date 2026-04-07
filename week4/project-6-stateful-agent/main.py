from agent import run_agent, get_history

THREAD_ID = "session-1"

DEMO_QUESTIONS = [
    "Research Kubernetes for me",
    "Now research Docker Swarm",
    "Compare what you found about Kubernetes vs Docker Swarm",
    "Summarize everything we discussed in this conversation",
]


def run_demo():
    print("=" * 60)
    print("Project 6 — Stateful Conversation Agent (Demo)")
    print("=" * 60)

    for question in DEMO_QUESTIONS:
        print(f"\nUSER: {question}")
        print("-" * 40)
        answer = run_agent(question, thread_id=THREAD_ID)
        print(f"AGENT: {answer}")

    print("\n" + "=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)
    for turn in get_history(THREAD_ID):
        label = "USER" if turn["role"] == "user" else "AGENT"
        preview = turn["content"][:120].replace("\n", " ")
        print(f"[{label}] {preview}...")


def run_interactive():
    print("=" * 60)
    print("Project 6 — Stateful Conversation Agent (Interactive)")
    print("Type 'history' to see conversation, 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "history":
            for turn in get_history(THREAD_ID):
                label = "YOU  " if turn["role"] == "user" else "AGENT"
                preview = turn["content"][:100].replace("\n", " ")
                print(f"  [{label}] {preview}")
            continue

        answer = run_agent(user_input, thread_id=THREAD_ID)
        print(f"\nAGENT: {answer}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        run_demo()
