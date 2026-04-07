from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import run_agent, get_history, clear_history

app = FastAPI(title="Agent API", version="1.0.0")


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    session_id: str


# ── Endpoints ────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agent/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = run_agent(request.question, session_id=request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/research", response_model=AskResponse)
def research(request: AskRequest):
    research_question = f"Please research and provide a detailed answer: {request.question}"
    try:
        result = run_agent(research_question, session_id=request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/history")
def history(session_id: str = "default"):
    return {"session_id": session_id, "history": get_history(session_id)}


@app.delete("/agent/history")
def delete_history(session_id: str = "default"):
    clear_history(session_id)
    return {"session_id": session_id, "cleared": True}
