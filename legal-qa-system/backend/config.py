import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"
METADATA_CSV = DATA_DIR / "legal_documents_100.csv"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

# Embedding model (small + fast, runs on CPU)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Hybrid search
VECTOR_TOP_K = 15
KEYWORD_TOP_K = 15
FINAL_TOP_K = 5
RRF_K = 60  # reciprocal rank fusion constant

# LLM (Anthropic) — set ANTHROPIC_API_KEY as an environment variable, never hardcode it
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_ANSWER_TOKENS = 1200
