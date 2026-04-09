from agent import run_with_approval


def run_demo():
    print("=" * 60)
    print("Project 7 — Human-in-the-Loop Agent (Demo)")
    print("=" * 60)

    scenarios = [
        {
            "label": "Scenario 1 — User APPROVES the save",
            "question": "Research Kafka and save a report to kafka_report.md",
            "thread_id": "demo-approve",
        },
        {
            "label": "Scenario 2 — User REJECTS the save",
            "question": "Research MLOps and save a report to mlops_report.md",
            "thread_id": "demo-reject",
        },
    ]

    for scenario in scenarios:
        print(f"\n{'─' * 60}")
        print(f"{scenario['label']}")
        print(f"{'─' * 60}")
        print(f"USER: {scenario['question']}")
        answer = run_with_approval(scenario["question"], scenario["thread_id"])
        print(f"\nAGENT: {answer}")


def run_interactive():
    print("=" * 60)
    print("Project 7 — Human-in-the-Loop Agent (Interactive)")
    print("The agent will ask for approval before saving any files.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    thread_id = "interactive-session"

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

        answer = run_with_approval(user_input, thread_id)
        print(f"\nAGENT: {answer}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        run_demo()
