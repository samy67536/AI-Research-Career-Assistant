# backend/main.py
import os, sys, uuid, shutil
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from config import settings
from database.connection import init_db
from services.llm_factory import get_llm_provider, set_runtime_config, validate_api_key
from services.rag_pipeline import RAGPipeline
from services.resume_analyzer import ResumeAnalyzer
from services.document_processor import DocumentProcessor

app = FastAPI(
    title=settings.app_name,
    description="RAG-based platform for Research Paper Analysis and Resume Evaluation",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global services (re-created when provider changes)
_rag    = None
_resume = None
_docproc = DocumentProcessor()


def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag


def get_resume_svc() -> ResumeAnalyzer:
    global _resume
    if _resume is None:
        _resume = ResumeAnalyzer()
    return _resume


@app.on_event("startup")
async def startup():
    await init_db()
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.vector_dir, exist_ok=True)
    print(f"[Server] Started | Provider: {settings.llm_provider}")


# ── Health ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "app": settings.app_name, "docs": "/docs"}


# ── Settings & Key Management ─────────────────────────────────────────
class KeyConfig(BaseModel):
    provider: str
    api_key:  str
    model:    Optional[str] = ""


@app.post("/settings/configure", tags=["Settings"])
def configure_llm(req: KeyConfig):
    """Set API key and provider at runtime from the UI."""
    global _rag, _resume
    try:
        set_runtime_config(req.provider, req.api_key, req.model or "")
        _rag    = RAGPipeline(provider=req.provider)
        _resume = ResumeAnalyzer(provider=req.provider)
        return {"success": True, "provider": req.provider, "message": f"Provider set to {req.provider}"}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/settings/validate-key", tags=["Settings"])
def check_key(req: KeyConfig):
    """Validate an API key with a test request."""
    result = validate_api_key(req.provider, req.api_key)
    if not result["valid"]:
        raise HTTPException(400, result["message"])
    return result


@app.get("/settings/providers", tags=["Settings"])
def list_providers():
    from services.llm_factory import _runtime_keys, _runtime_provider
    return {
        "available": ["claude", "openai", "gemini", "deepseek"],
        "current":   _runtime_provider or settings.llm_provider,
        "models": {
            "claude":   ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
            "openai":   ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            "gemini":   ["gemini-1.5-flash", "gemini-1.5-pro"],
            "deepseek": ["deepseek-chat", "deepseek-coder"],
        }
    }


# ── Research Paper ────────────────────────────────────────────────────
@app.post("/research/upload", tags=["Research Paper"])
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")
    doc_id    = str(uuid.uuid4())
    file_path = os.path.join(settings.upload_dir, f"{doc_id}.pdf")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = get_rag().ingest_document(file_path, "pdf", doc_id, file.filename)
        return {"success": True, "doc_id": doc_id, **result}
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(500, str(e))


class AskRequest(BaseModel):
    doc_id:       str
    question:     str
    chat_history: Optional[List[dict]] = []


@app.post("/research/ask", tags=["Research Paper"])
def ask(req: AskRequest):
    try:
        return {"success": True, **get_rag().query(req.question, req.doc_id, chat_history=req.chat_history)}
    except Exception as e:
        raise HTTPException(500, str(e))


class DocRequest(BaseModel):
    doc_id:       str
    summary_type: Optional[str] = "structured"


@app.post("/research/summarize", tags=["Research Paper"])
def summarize(req: DocRequest):
    try:
        return {"success": True, **get_rag().summarize_document(req.doc_id, req.summary_type)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/research/key-findings", tags=["Research Paper"])
def findings(req: DocRequest):
    try:
        return {"success": True, **get_rag().extract_key_findings(req.doc_id)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/research/references", tags=["Research Paper"])
def references(req: DocRequest):
    try:
        return {"success": True, **get_rag().extract_references(req.doc_id)}
    except Exception as e:
        raise HTTPException(500, str(e))


class CompareRequest(BaseModel):
    doc_ids:   List[str]
    doc_names: List[str]


@app.post("/research/compare", tags=["Research Paper"])
def compare(req: CompareRequest):
    try:
        return {"success": True, **get_rag().compare_papers(req.doc_ids, req.doc_names)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Resume ────────────────────────────────────────────────────────────
@app.post("/resume/upload-analyze", tags=["Resume"])
async def analyze_resume(file: UploadFile = File(...), job_description: str = Form(default="")):
    fname = file.filename.lower()
    if not (fname.endswith(".pdf") or fname.endswith(".docx")):
        raise HTTPException(400, "Only PDF or DOCX accepted.")
    ext       = "pdf" if fname.endswith(".pdf") else "docx"
    doc_id    = str(uuid.uuid4())
    file_path = os.path.join(settings.upload_dir, f"{doc_id}.{ext}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        text, meta = _docproc.extract_text(file_path, ext)
        analysis   = get_resume_svc().analyze(text, job_description)
        return {"success": True, "doc_id": doc_id, "filename": file.filename,
                "word_count": meta.get("word_count", 0), **analysis}
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, reload=True)
