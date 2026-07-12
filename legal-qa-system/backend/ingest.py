"""
Ingestion pipeline.

Drop source documents (.pdf or .txt) into data/documents/, optionally with
a legal_documents_100.csv metadata file (doc title, category, url) in data/,
then run:

    python backend/ingest.py

This will:
  1. Parse each document, preserving page numbers (PDF) or treating .txt as one page.
  2. Chunk text into overlapping windows, tagging each chunk with {doc_id, page, section}.
  3. Embed all chunks (sentence-transformers) and build a FAISS vector index.
  4. Build a BM25 keyword index over the same chunks.
  5. Persist everything to data/index/ so the API can load it instantly at startup.
"""
import json
import pickle
import re
import sys
import csv
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import DOCS_DIR, INDEX_DIR, METADATA_CSV, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, EMBEDDING_MODEL


def load_metadata():
    """Optional CSV with columns: ID,Category,Document Title / Citation,Description,Source / Publisher,URL"""
    meta = {}
    if METADATA_CSV.exists():
        with open(METADATA_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                title = row.get("Document Title / Citation", "").strip()
                if title:
                    meta[title.lower()] = row
    return meta


def extract_pdf_pages(path: Path):
    """Returns list of (page_number, text) tuples."""
    import fitz  # PyMuPDF
    pages = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((i, text))
    return pages


def extract_txt_pages(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(1, text)]


def chunk_text(text: str, size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def guess_section(text: str):
    """Best-effort section/heading detector for statutes and judgments."""
    m = re.search(r"(§+\s?\d+[A-Za-z]?(\(\w+\))?)", text)
    if m:
        return m.group(0)
    m = re.search(r"\b(Section|Sec\.|Article)\s+[\dIVXLC]+", text, re.IGNORECASE)
    if m:
        return m.group(0)
    return None


def build_index():
    from sentence_transformers import SentenceTransformer
    import faiss
    from rank_bm25 import BM25Okapi

    metadata = load_metadata()
    all_chunks = []  # list of dicts: {doc_id, doc_title, page, section, text, url, category}

    doc_files = sorted([p for p in DOCS_DIR.glob("*") if p.suffix.lower() in (".pdf", ".txt")])
    if not doc_files:
        print(f"No documents found in {DOCS_DIR}. Add .pdf or .txt files and rerun.")
        return

    for path in doc_files:
        doc_id = path.stem
        meta_row = metadata.get(doc_id.lower(), {})
        doc_title = meta_row.get("Document Title / Citation", doc_id)
        url = meta_row.get("URL", "")
        category = meta_row.get("Category", "")

        pages = extract_pdf_pages(path) if path.suffix.lower() == ".pdf" else extract_txt_pages(path)

        for page_num, page_text in pages:
            for chunk in chunk_text(page_text):
                all_chunks.append({
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "page": page_num,
                    "section": guess_section(chunk),
                    "text": chunk,
                    "url": url,
                    "category": category,
                })

    if not all_chunks:
        print("No extractable text found in the provided documents.")
        return

    print(f"Parsed {len(doc_files)} documents into {len(all_chunks)} chunks.")

    print("Embedding chunks (this runs locally, no API calls)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)  # cosine sim via normalized inner product
    faiss_index.add(embeddings)

    print("Building BM25 keyword index...")
    tokenized = [re.findall(r"[a-zA-Z0-9§]+", t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(INDEX_DIR / "vector.index"))
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    print(f"Index built: {len(all_chunks)} chunks from {len(doc_files)} documents.")
    print(f"Saved to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
