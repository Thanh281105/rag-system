from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

from src.rag_engine.agent import LegalAgent

router = APIRouter()

# ==============================
# Request / Response schema
# ==============================

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


# ==============================
# Load Agent (singleton)
# ==============================

DB_DIR = "data/vector_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

agent = LegalAgent(
    db_dir=DB_DIR,
    groq_api_key=GROQ_API_KEY
)


# ==============================
# Health check
# ==============================

@router.get("/")
def root():
    return {"message": "Legal RAG API running 🚀"}


# ==============================
# Ask question
# ==============================

@router.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        answer = agent.ask(request.question)

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )