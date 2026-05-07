"""
SWS AI RAG Chatbot — FastAPI Backend
--------------------------------------
Endpoints:
  POST /api/chat   — accepts { question } → returns { answer, sources }
  GET  /api/health — health check
  GET  /api/docs-list — list ingested documents

Run: uvicorn main:app --reload --port 8000
"""

import os
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from rag_chain import get_rag_chain

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "SWS AI RAG Chatbot API",
    description = "Retrieval-Augmented Generation over SWS AI company documents",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Load RAG chain once at startup ────────────────────────────────────────────
print("🔄 Loading RAG chain...")
rag_chain = get_rag_chain()
print("✅ RAG chain ready!")


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer  : str
    sources : List[str]
    duration: float         # seconds taken


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "claude-3-5-sonnet-20241022"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.time()

    try:
        result = rag_chain.invoke({"query": question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG chain error: {str(e)}")

    # Extract unique source document names
    source_docs = result.get("source_documents", [])
    sources = sorted(set(
        doc.metadata.get("doc_title") or doc.metadata.get("source", "Unknown Document")
        for doc in source_docs
    ))

    answer   = result.get("result", "I don't have that information in the company documents.")
    duration = round(time.time() - t0, 2)

    return ChatResponse(answer=answer, sources=sources, duration=duration)


@app.get("/api/docs-list")
async def list_docs():
    """Return list of ingested document names."""
    chroma_dir = Path(__file__).parent / "chroma_db"
    if not chroma_dir.exists():
        return {"documents": [], "message": "No documents ingested yet."}

    docs_dir = Path(__file__).parent.parent / "documents"
    pdf_files = [f.name for f in docs_dir.glob("*.pdf")] if docs_dir.exists() else []
    return {"documents": pdf_files, "count": len(pdf_files)}


# ── Serve frontend ────────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_ui():
        return FileResponse(str(frontend_dir / "index.html"))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
