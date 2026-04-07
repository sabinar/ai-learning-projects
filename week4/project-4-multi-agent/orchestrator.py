import os
import time
from dotenv import load_dotenv
from agents import run_research_agent, run_writer_agent

load_dotenv()

def run_multi_agent_pipeline(topic: str) -> str:
    """
    Orchestrate multiple agents to research and write a report.
    
    Pipeline:
    1. Planner — breaks topic into research questions
    2. Research Agent — gathers information using tools
    3. Writer Agent — transforms research into polished report
    """
    print("\n" + "="*60)
    print("MULTI-AGENT PIPELINE STARTING")
    print("="*60)
    print(f"Topic: {topic}")
    start_time = time.time()

    # ── Stage 1 — Plan ───────────────────────────────────
    print("\n📋 STAGE 1 — PLANNING")
    print("-"*40)
    print(f"Breaking down topic: {topic}")
    
    # For this topic, define what to research
    research_questions = [
        f"Core concepts and architecture of {topic}",
        f"Strengths and weaknesses of {topic}",
        f"Real-world use cases for {topic}",
    ]
    
    print("Research questions:")
    for i, q in enumerate(research_questions, 1):
        print(f"  {i}. {q}")

    # ── Stage 2 — Research ───────────────────────────────
    print("\n🔍 STAGE 2 — RESEARCH")
    print("-"*40)

    all_research = []
    for question in research_questions:
        research = run_research_agent(question)
        all_research.append(f"### {question}\n{research}")

    combined_research = "\n\n".join(all_research)
    print(f"\n📊 Total research gathered: {len(combined_research)} chars")

    # ── Stage 3 — Write ──────────────────────────────────
    print("\n✍️  STAGE 3 — WRITING")
    print("-"*40)

    report = run_writer_agent(topic, combined_research)

    # ── Stage 4 — Save ───────────────────────────────────
    print("\n💾 STAGE 4 — SAVING")
    print("-"*40)

    filename = f"report_{topic.lower().replace(' ', '_')}.md"
    with open(filename, "w") as f:
        f.write(report)
    print(f"Report saved to {filename}")

    # ── Summary ──────────────────────────────────────────
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Topic:       {topic}")
    print(f"Research:    {len(combined_research)} chars gathered")
    print(f"Report:      {len(report)} chars written")
    print(f"Saved to:    {filename}")
    print(f"Total time:  {total_time:.1f} seconds")

    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(report)

    return report

def run_comparison_pipeline(tech1: str, tech2: str) -> str:
    """
    Specialized pipeline for technology comparisons.
    Runs research agents on both technologies in parallel concept,
    then combines for a comparison report.
    """
    print("\n" + "="*60)
    print("COMPARISON PIPELINE STARTING")
    print("="*60)
    print(f"Comparing: {tech1} vs {tech2}")
    start_time = time.time()

    # Research both technologies
    print(f"\n🔍 Researching {tech1}...")
    research1 = run_research_agent(tech1)

    print(f"\n🔍 Researching {tech2}...")
    research2 = run_research_agent(tech2)

    # Combine research
    combined = f"""# Research on {tech1}
{research1}

# Research on {tech2}
{research2}"""

    # Write comparison report
    comparison_topic = f"{tech1} vs {tech2} — Detailed Comparison"
    report = run_writer_agent(comparison_topic, combined)

    # Save report
    filename = f"comparison_{tech1.lower().replace(' ', '_')}_vs_{tech2.lower().replace(' ', '_')}.md"
    with open(filename, "w") as f:
        f.write(report)

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Saved to:   {filename}")
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(report)

    return report

if __name__ == "__main__":
    # Pipeline 1 — Single topic deep dive
    print("\n🚀 PIPELINE 1 — SINGLE TOPIC")
    run_multi_agent_pipeline("Kubernetes")

    print("\n\n")

    # Pipeline 2 — Technology comparison
    print("\n🚀 PIPELINE 2 — COMPARISON")
    run_comparison_pipeline("Kubernetes", "Docker Swarm")