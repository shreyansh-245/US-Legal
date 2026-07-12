import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import ANTHROPIC_API_KEY
from models import AskRequest, AskResponse, Citation, DocumentInfo, HealthResponse
from retriever import retriever
from llm import generate_answer

app = FastAPI(title="Legal Tax & Law Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health():
    chunk_count = len(retriever.chunks) if retriever.ready else 0
    doc_count = len(retriever.document_summary()) if retriever.ready else 0
    return HealthResponse(
        status="ready" if retriever.ready else "no_index_found",
        documents_indexed=doc_count,
        chunks_indexed=chunk_count,
    )


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents():
    if not retriever.ready:
        raise HTTPException(503, "Index not built yet. Run backend/ingest.py first.")
    return retriever.document_summary()


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not retriever.ready:
        raise HTTPException(503, "Index not built yet. Run backend/ingest.py first.")
    if not request.question.strip():
        raise HTTPException(400, "Question must not be empty.")

    hits = retriever.search(request.question, top_k=request.top_k)
    answer = generate_answer(request.question, hits)

    citations = [
        Citation(
            doc_id=h["doc_id"],
            doc_title=h["doc_title"],
            page=h.get("page"),
            section=h.get("section"),
            url=h.get("url"),
            snippet=h["text"][:300] + ("..." if len(h["text"]) > 300 else ""),
            score=h["score"],
        )
        for h in hits
    ]

    return AskResponse(
        answer=answer,
        citations=citations,
        mode="llm" if ANTHROPIC_API_KEY else "retrieval_only",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
