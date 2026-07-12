import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import INDEX_DIR, EMBEDDING_MODEL, VECTOR_TOP_K, KEYWORD_TOP_K, FINAL_TOP_K, RRF_K


class HybridRetriever:
    """
    Loads the persisted vector index (FAISS), keyword index (BM25), and chunk
    metadata built by ingest.py, and answers queries with reciprocal-rank-fusion
    (RRF) combining semantic and exact-keyword results.
    """

    def __init__(self):
        self.ready = False
        self._load()

    def _load(self):
        chunks_path = INDEX_DIR / "chunks.json"
        vec_path = INDEX_DIR / "vector.index"
        bm25_path = INDEX_DIR / "bm25.pkl"

        if not (chunks_path.exists() and vec_path.exists() and bm25_path.exists()):
            self.ready = False
            return

        import faiss
        from sentence_transformers import SentenceTransformer

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.vector_index = faiss.read_index(str(vec_path))

        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.ready = True

    def _vector_search(self, query: str, k: int):
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        scores, idxs = self.vector_index.search(q_emb, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def _keyword_search(self, query: str, k: int):
        tokens = re.findall(r"[a-zA-Z0-9§]+", query.lower())
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def search(self, query: str, top_k: int = FINAL_TOP_K):
        """Reciprocal Rank Fusion of vector + keyword result rankings."""
        if not self.ready:
            return []

        vector_hits = self._vector_search(query, VECTOR_TOP_K)
        keyword_hits = self._keyword_search(query, KEYWORD_TOP_K)

        rrf_scores = {}
        for rank, (idx, _) in enumerate(vector_hits):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(keyword_hits):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            results.append({
                "doc_id": chunk["doc_id"],
                "doc_title": chunk["doc_title"],
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "url": chunk.get("url"),
                "text": chunk["text"],
                "score": round(score, 4),
            })
        return results

    def document_summary(self):
        if not self.ready:
            return []
        by_doc = {}
        for c in self.chunks:
            key = c["doc_id"]
            if key not in by_doc:
                by_doc[key] = {"doc_id": key, "title": c["doc_title"], "category": c.get("category"),
                               "url": c.get("url"), "chunk_count": 0}
            by_doc[key]["chunk_count"] += 1
        return list(by_doc.values())


# Singleton instance used by the API
retriever = HybridRetriever()
