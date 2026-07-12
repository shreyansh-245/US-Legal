from pydantic import BaseModel
from typing import List, Optional


class Citation(BaseModel):
    doc_id: str
    doc_title: str
    page: Optional[int] = None
    section: Optional[str] = None
    url: Optional[str] = None
    snippet: str
    score: float


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    mode: str  # "llm" or "retrieval_only"


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    category: Optional[str] = None
    url: Optional[str] = None
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_indexed: int
