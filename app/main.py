from fastapi import FastAPI, HTTPException
from app.models import UserMessage, BotResponse
from app.chat import get_chat_response, detect_intent
from app.guardrails import run_guardrails
import uuid

app = FastAPI(
    title="Retail Support Chatbot",
    description="AI-powered customer support for retail & e-commerce",
    version="3.0.0"
)

# In-memory session store — dict of session_id → conversation history
sessions: dict[str, list[dict]] = {}


@app.get("/")
def root():
    return {"status": "Retail Support Bot v3 is running 🛍️"}


@app.post("/chat", response_model=BotResponse)
def chat(user_input: UserMessage):

    # 1. Resolve session
    session_id = user_input.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    history = sessions[session_id]

    # 2. Run guardrails BEFORE anything else
    #    If a check triggers, we return immediately — no LLM call wasted
    guard_result = run_guardrails(user_input.message, history)

    if guard_result["action"] in ("escalate", "out_of_scope"):
        # Log the user message even if we escalate — useful for audit
        history.append({"role": "user", "content": user_input.message})
        history.append({"role": "assistant", "content": guard_result["message"]})
        sessions[session_id] = history

        return BotResponse(
            reply=guard_result["message"],
            session_id=session_id,
            intent=detect_intent(user_input.message),
            action=guard_result["action"]
        )

    # 3. Detect intent (only if not escalated)
    intent = detect_intent(user_input.message)

    # 4. Add user message to history
    history.append({"role": "user", "content": user_input.message})

    # 5. Get Claude's response with RAG context
    reply = get_chat_response(history)

    # 6. Save assistant reply to history
    history.append({"role": "assistant", "content": reply})
    sessions[session_id] = history

    return BotResponse(
        reply=reply,
        session_id=session_id,
        intent=intent,
        action="continue"
    )


@app.delete("/chat/reset/{session_id}")
def reset_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions.pop(session_id)
    return {"status": f"Session {session_id} cleared ✅"}


@app.get("/chat/history/{session_id}")
def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": sessions[session_id]}


@app.get("/knowledge/search")
def search_knowledge(query: str):
    from app.knowledge_base import knowledge_base
    results = knowledge_base.search(query, top_k=3)
    return {"query": query, "results": results}