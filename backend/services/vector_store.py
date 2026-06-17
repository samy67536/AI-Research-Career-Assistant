# backend/services/vector_store.py
"""
Vector Store Service
Uses: Sentence Transformers (all-MiniLM-L6-v2) + FAISS
Handles: embedding generation, index building, similarity search, persistence
"""
import os
import sys
import json
import pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings
from typing import List, Dict


class VectorStore:

    def __init__(self):
        self.model      = None   # loaded lazily
        self.index      = None
        self.chunks     = []
        self.metadata   = []
        self.model_name = settings.embedding_model

    # ── Lazy model loader ─────────────────────────────────────────────

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    # ── Build index from chunks ───────────────────────────────────────

    def build_index(self, chunks: List[Dict]) -> int:
        """
        Build FAISS index from list of chunk dicts.
        Each chunk: {text, chunk_index, word_count, metadata}
        Returns: number of vectors indexed
        """
        import faiss

        texts = [c["text"] for c in chunks]
        self.chunks   = texts
        self.metadata = [c.get("metadata", {}) | {"chunk_index": c.get("chunk_index", i)}
                         for i, c in enumerate(chunks)]

        model = self._get_model()
        print(f"Encoding {len(texts)} chunks...")
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                                  convert_to_numpy=True)

        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

        self.index = faiss.IndexFlatIP(dim)   # Inner Product = cosine on normalized
        self.index.add(embeddings)

        print(f"FAISS index built with {self.index.ntotal} vectors (dim={dim})")
        return self.index.ntotal

    # ── Similarity search ─────────────────────────────────────────────

    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        Search for top-k most similar chunks to the query.
        Returns list of {text, score, metadata, chunk_index}
        """
        if self.index is None:
            raise RuntimeError("Vector index not built. Call build_index() first.")

        import faiss
        k = k or settings.top_k_results

        model = self._get_model()
        q_emb = model.encode([query], convert_to_numpy=True).astype("float32")

        # Normalize query
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)

        scores, indices = self.index.search(q_emb, min(k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < 0.1:   # filter irrelevant
                continue
            results.append({
                "text":        self.chunks[idx],
                "score":       float(score),
                "chunk_index": idx,
                "metadata":    self.metadata[idx] if idx < len(self.metadata) else {}
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, doc_id: str) -> str:
        """Save FAISS index and chunks to disk. Returns saved directory path."""
        import faiss

        save_dir = os.path.join(settings.vector_dir, doc_id)
        os.makedirs(save_dir, exist_ok=True)

        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        with open(os.path.join(save_dir, "data.pkl"), "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)

        print(f"Vector store saved to: {save_dir}")
        return save_dir

    def load(self, doc_id: str) -> bool:
        """Load FAISS index and chunks from disk."""
        import faiss

        save_dir = os.path.join(settings.vector_dir, doc_id)
        index_path = os.path.join(save_dir, "index.faiss")
        data_path  = os.path.join(save_dir, "data.pkl")

        if not os.path.exists(index_path):
            return False

        self.index = faiss.read_index(index_path)
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.chunks   = data["chunks"]
        self.metadata = data["metadata"]

        print(f"Loaded vector store: {self.index.ntotal} vectors from {save_dir}")
        return True

    def exists(self, doc_id: str) -> bool:
        """Check if a vector index exists for the given doc_id."""
        return os.path.exists(os.path.join(settings.vector_dir, doc_id, "index.faiss"))

    # ── Context builder ───────────────────────────────────────────────

    def build_context(self, results: List[Dict], max_chars: int = 4000) -> str:
        """Assemble retrieved chunks into a context string for the LLM."""
        context_parts = []
        total_chars   = 0

        for i, r in enumerate(results):
            chunk_text = r["text"]
            meta       = r.get("metadata", {})
            source_tag = f"[Chunk {i+1}"
            if meta.get("page_num"):
                source_tag += f" | Page {meta['page_num']}"
            if meta.get("filename"):
                source_tag += f" | {meta['filename']}"
            source_tag += "]"

            entry = f"{source_tag}\n{chunk_text}"

            if total_chars + len(entry) > max_chars:
                break

            context_parts.append(entry)
            total_chars += len(entry)

        return "\n\n---\n\n".join(context_parts)
