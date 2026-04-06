# Project 1 — LLM Chains

Multi-step reasoning pipeline using LangChain and Claude.
Chains three LLM calls together: summarize → extract topics → generate questions.

## Pipeline
Input text
↓
Step 1: Summarize (3-4 sentences)
↓
Step 2: Extract key topics (bullet list)
↓
Step 3: Generate questions (3 deep questions)

## Key Concepts
- LangChain pipe syntax: prompt | llm | parser
- Streaming vs batch responses
- Temperature control — 0.0 for deterministic, 0.9 for creative
- ChatPromptTemplate for structured prompts

## How to Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-anthropic python-dotenv
echo "ANTHROPIC_API_KEY=your_key" > .env
python chain.py
```

## Tech Stack
- LangChain — chain orchestration
- Claude Sonnet — LLM
- Anthropic API — model provider
