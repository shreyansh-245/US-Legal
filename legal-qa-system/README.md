MAKE SURE TO DOWNLOAD PYTHON 3.12.0

# Legal Tax & Law Q&A System

A retrieval-augmented Q&A system for legal/tax documents. Every answer is generated
strictly from your indexed source documents and includes inline citations
(document + page number) that link back to the retrieved passage.

## Architecture

```
PDF / TXT documents
      │
      ▼
backend/ingest.py        parses text, preserves page numbers, chunks, embeds
      │
      ▼
data/index/               FAISS vector index + BM25 keyword index + chunk metadata
      │
      ▼
backend/retriever.py      hybrid search (reciprocal rank fusion of vector + keyword)
      │
      ▼
backend/llm.py            Claude generates a cited answer from the retrieved chunks
      │
      ▼
backend/main.py (FastAPI) /api/ask endpoint
      │
      ▼
frontend/index.html       chat UI with citations and source links
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your documents**
   Drop `.pdf` or `.txt` files into `data/documents/`. A sample file
   (`IRC-Section-162.txt`) is included so you can test the pipeline immediately.

   Optionally, edit `data/legal_documents_100.csv` (or provide your own) with
   columns `Document Title / Citation`, `Category`, `URL` — the filename
   (without extension) should match a title in this file (case-insensitive)
   so citations get a clickable source URL and category tag.

3. **Set your Anthropic API key** (optional but recommended)
   ```bash
   cp .env.example .env
   # edit .env and set ANTHROPIC_API_KEY
   export $(cat .env | xargs)
   ```
   Without a key, the system still runs — `/api/ask` returns the raw
   top-ranked passages instead of an LLM-synthesized answer, which is useful
   for testing retrieval quality on its own.

4. **Build the index**
   ```bash
   python backend/ingest.py
   ```

5. **Start the API**
   ```bash
   python backend/main.py
   # or: uvicorn backend.main:app --reload --port 8000
   ```

6. **Open the UI**
   Open `frontend/index.html` directly in a browser (or serve it with any
   static file server). It talks to the API at `http://localhost:8000`.

Or run all of the above in one go:
```bash
bash scripts/run.sh
```

## API

### `POST /api/ask`
```json
{ "question": "What is the standard for an ordinary and necessary business expense?", "top_k": 5 }
```
Returns:
```json
{
  "answer": "An expense is deductible under IRC § 162 if it is...",
  "citations": [
    { "doc_title": "IRC Section 162", "page": 1, "url": "...", "snippet": "...", "score": 0.83 }
  ],
  "mode": "llm"
}
```

### `GET /api/documents`
Lists every indexed document and its chunk count.

### `GET /api/health`
Reports whether the index has been built and how many documents/chunks it contains.

## Evaluating retrieval quality (Golden Set)

To measure retrieval accuracy and faithfulness against a golden set of
`{query, ground_truth_answer, source_document}` triples, call `/api/ask` for
each query and check:
- **Retrieval accuracy**: does `citations[i].doc_id` match the expected source document?
- **Faithfulness**: does the LLM answer's claims trace back to the cited snippet
  text (spot-check manually, or use a second LLM call as a judge)?

## Notes on scaling

This reference implementation uses FAISS `IndexFlatIP` (exact search) and an
in-memory BM25 index, which is appropriate for up to a few thousand documents.
For a larger corpus, swap in a managed vector DB (Qdrant, Weaviate, pgvector)
and Elasticsearch for keyword search — the `HybridRetriever` interface in
`backend/retriever.py` is written so you can swap the underlying store without
touching `main.py`.

## License

Use and modify freely for your own project.
