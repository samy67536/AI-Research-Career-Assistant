# backend/services/rag_pipeline.py
"""
RAG Pipeline
Orchestrates: Document ingestion → embedding → storage → retrieval → LLM response
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore
from services.llm_factory import get_llm_provider
from config import settings
from typing import Dict, List, Optional


class RAGPipeline:

    def __init__(self, provider: str = None):
        self.processor = DocumentProcessor()
        self.llm       = get_llm_provider(provider)
        self._stores: Dict[str, VectorStore] = {}   # doc_id → VectorStore

    # ── Ingestion ─────────────────────────────────────────────────────

    def ingest_document(self, file_path: str, file_type: str, doc_id: str,
                        filename: str = "") -> Dict:
        """
        Full ingestion pipeline:
        1. Extract text
        2. Chunk text
        3. Build vector index
        4. Persist to disk
        """
        print(f"[RAG] Ingesting: {filename} ({file_type})")

        # 1. Extract
        text, meta = self.processor.extract_text(file_path, file_type)
        meta["filename"] = filename or os.path.basename(file_path)

        # 2. Chunk
        chunks = self.processor.chunk_text(text, metadata=meta)
        print(f"[RAG] Created {len(chunks)} chunks")

        # 3. Build vector index
        store = VectorStore()
        num_vectors = store.build_index(chunks)

        # 4. Persist
        store.save(doc_id)
        self._stores[doc_id] = store

        return {
            "doc_id":      doc_id,
            "filename":    meta["filename"],
            "page_count":  meta.get("page_count", 0),
            "word_count":  meta.get("word_count", 0),
            "chunk_count": len(chunks),
            "vectors":     num_vectors
        }

    # ── Query ─────────────────────────────────────────────────────────

    def query(self, question: str, doc_id: str,
              k: int = None, chat_history: List[Dict] = None) -> Dict:
        """
        Full RAG query pipeline:
        1. Load vector store
        2. Retrieve relevant chunks
        3. Build context
        4. Generate LLM response
        """
        store = self._get_store(doc_id)
        k     = k or settings.top_k_results

        # 1. Retrieve
        results = store.search(question, k=k)
        if not results:
            return {
                "answer":  "I could not find relevant information in the uploaded document. Please try rephrasing your question.",
                "sources": [],
                "chunks_used": 0
            }

        # 2. Build context
        context = store.build_context(results)

        # 3. Build history string (optional)
        history_str = ""
        if chat_history:
            recent = chat_history[-4:]   # last 2 turns
            history_str = "\n".join([
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in recent
            ])

        # 4. Prompt + LLM
        system_prompt = (
            "You are an expert AI research assistant. "
            "Answer questions ONLY based on the provided document context. "
            "If the answer is not in the context, say: 'This information is not available in the uploaded document.' "
            "Always cite specific chunk numbers (e.g. [Chunk 1]) when referencing content. "
            "Be concise, accurate, and well-structured."
        )

        user_prompt = ""
        if history_str:
            user_prompt += f"Previous conversation:\n{history_str}\n\n"
        user_prompt += f"Document Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1500,
            temperature=0.2
        )

        sources = [
            {
                "chunk_index": r["chunk_index"],
                "score":       round(r["score"], 3),
                "preview":     r["text"][:120] + "...",
                "page":        r["metadata"].get("page_num", "N/A")
            }
            for r in results[:3]
        ]

        return {
            "answer":      response.text,
            "sources":     sources,
            "chunks_used": len(results),
            "tokens_used": response.tokens_used
        }

    # ── Summary ───────────────────────────────────────────────────────

    def summarize_document(self, doc_id: str, summary_type: str = "structured") -> Dict:
        """
        Generate a document summary.
        summary_type: 'brief' | 'structured' | 'beginner'
        """
        store = self._get_store(doc_id)

        # Get a broad sample of the document
        sample_results = store.search("main objective methodology results conclusion", k=8)
        context = store.build_context(sample_results, max_chars=5000)

        prompts = {
            "brief": (
                "Summarize this research paper in 3-5 sentences covering "
                "the main objective, approach, and key finding."
            ),
            "structured": (
                "Provide a structured summary of this research paper with these sections:\n"
                "**Objective**: What problem does it solve?\n"
                "**Methodology**: What approach/techniques are used?\n"
                "**Key Findings**: What are the main results?\n"
                "**Contributions**: What is novel/unique?\n"
                "**Limitations**: What are the acknowledged limitations?\n"
                "**Future Work**: What future directions are mentioned?"
            ),
            "beginner": (
                "Explain this research paper in simple language for a beginner with no domain expertise. "
                "Avoid jargon. Use analogies where helpful. Cover: what it's about, why it matters, "
                "what was done, and what was found. Use plain English."
            )
        }

        prompt = prompts.get(summary_type, prompts["structured"])

        system_prompt = (
            "You are an expert at summarizing academic research papers clearly and accurately. "
            "Base your summary entirely on the provided context."
        )
        user_prompt = f"Document Context:\n{context}\n\n{prompt}"

        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1500,
            temperature=0.3
        )

        return {
            "summary":      response.text,
            "summary_type": summary_type,
            "tokens_used":  response.tokens_used
        }

    # ── Key Findings ──────────────────────────────────────────────────

    def extract_key_findings(self, doc_id: str) -> Dict:
        store = self._get_store(doc_id)
        results = store.search("results findings conclusions contributions experiments", k=7)
        context = store.build_context(results, max_chars=4500)

        system_prompt = "You are an expert research analyst. Extract structured insights from academic papers."
        user_prompt = (
            f"Document Context:\n{context}\n\n"
            "Extract and list the following from this research paper:\n"
            "1. **Research Question/Hypothesis** (1-2 sentences)\n"
            "2. **Key Results** (bullet points with metrics/numbers where available)\n"
            "3. **Main Conclusions** (bullet points)\n"
            "4. **Novel Contributions** (what is new/unique)\n"
            "5. **Datasets/Benchmarks Used** (if mentioned)\n"
            "6. **Performance Metrics** (accuracy, F1, BLEU, etc.)\n"
            "Format clearly using markdown."
        )

        response = self.llm.generate(system_prompt, user_prompt, max_tokens=1200, temperature=0.2)
        return {"findings": response.text}

    # ── Compare Papers ────────────────────────────────────────────────

    def compare_papers(self, doc_ids: List[str], doc_names: List[str]) -> Dict:
        """Compare multiple research papers."""
        if len(doc_ids) < 2:
            raise ValueError("Need at least 2 documents to compare.")

        summaries = []
        for doc_id, name in zip(doc_ids, doc_names):
            store = self._get_store(doc_id)
            results = store.search("objective methodology results contributions", k=5)
            context = store.build_context(results, max_chars=2000)
            summaries.append(f"=== PAPER: {name} ===\n{context}")

        combined = "\n\n".join(summaries)

        system_prompt = "You are an expert research analyst specializing in comparative analysis."
        user_prompt = (
            f"Compare these {len(doc_ids)} research papers:\n\n{combined}\n\n"
            "Provide a structured comparison covering:\n"
            "| Aspect | " + " | ".join(doc_names) + " |\n"
            "1. Research Problem\n2. Methodology\n3. Dataset Used\n"
            "4. Key Results\n5. Novelty\n6. Limitations\n\n"
            "Then write a paragraph summarizing: which paper is stronger and why, "
            "and how they complement each other."
        )

        response = self.llm.generate(system_prompt, user_prompt, max_tokens=2000, temperature=0.3)
        return {"comparison": response.text, "papers_compared": doc_names}

    # ── Extract References ────────────────────────────────────────────

    def extract_references(self, doc_id: str) -> Dict:
        store = self._get_store(doc_id)
        results = store.search("references bibliography works cited", k=6)
        context = store.build_context(results, max_chars=4000)

        system_prompt = "You are an expert at extracting and formatting academic references."
        user_prompt = (
            f"Document Context:\n{context}\n\n"
            "Extract all references/citations from this document. "
            "Format each in APA style as a numbered list. "
            "If you find in-text citations like [1] or (Author, Year), list them too."
        )

        response = self.llm.generate(system_prompt, user_prompt, max_tokens=1500, temperature=0.1)
        return {"references": response.text}

    # ── Internal helpers ──────────────────────────────────────────────

    def _get_store(self, doc_id: str) -> VectorStore:
        if doc_id not in self._stores:
            store = VectorStore()
            if not store.load(doc_id):
                raise RuntimeError(f"No vector index found for doc_id: {doc_id}. Re-upload the document.")
            self._stores[doc_id] = store
        return self._stores[doc_id]
