# frontend/pages/01_Research_Assistant.py
import streamlit as st
import requests
import sys, os, uuid

# Fix sys.path so backend imports work in both HTTP and direct mode
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

st.set_page_config(page_title="Research Assistant", page_icon="📄", layout="wide")

st.markdown("""
<style>
.chat-user  { background:#E8F1FB; padding:0.9rem 1.1rem; border-radius:10px 10px 10px 2px; margin:0.5rem 0; }
.chat-bot   { background:#F0F7F0; padding:0.9rem 1.1rem; border-radius:10px 10px 2px 10px; margin:0.5rem 0; border-left: 3px solid #27AE60; }
.source-box { background:#FFF8E1; border-left:3px solid #F59E0B; padding:0.5rem 0.9rem; border-radius:4px; margin:0.25rem 0; font-size:0.82rem; }
.doc-card   { background:#F5F8FF; border:1px solid #C8D8F0; border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0; }
.status-ok  { color:#27AE60; font-weight:600; }
.status-err { color:#E74C3C; font-weight:600; }
</style>
""", unsafe_allow_html=True)

BACKEND = "http://localhost:8000"

st.title("📄 AI Research Paper Assistant")
st.caption("Upload PDFs · Q&A · Summarize · Compare · Extract Findings")

# ── Session init ──────────────────────────────────────────────────────
for k, v in {
    "r_docs": [], "r_active_id": "", "r_active_name": "",
    "r_chat": [], "r_last_summary": "", "r_last_findings": "",
    "r_last_refs": "", "r_comparison": ""
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def check_key():
    if not st.session_state.get("key_validated"):
        st.warning("⚠️ API key not configured. Go to the **Home** page and set your key in the sidebar.")
        return False
    return True


def call_backend(endpoint: str, method="post", json=None, files=None, data=None):
    url = f"{BACKEND}{endpoint}"
    try:
        if method == "post":
            resp = requests.post(url, json=json, files=files, data=data, timeout=120)
        else:
            resp = requests.get(url, timeout=30)

        if resp.status_code == 200:
            return resp.json(), None
        else:
            detail = resp.json().get("detail", resp.text) if resp.content else resp.text
            return None, f"Server error {resp.status_code}: {detail}"
    except requests.exceptions.ConnectionError:
        return None, "backend_offline"
    except Exception as e:
        return None, str(e)


def _get_backend_path():
    """Get absolute path to backend directory."""
    return os.path.join(_root, "backend")


def direct_upload(file_bytes: bytes, filename: str) -> tuple:
    """Upload and ingest a document using direct Python calls (no HTTP backend needed)."""
    try:
        backend_path = _get_backend_path()
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.llm_factory import set_runtime_config
        from services.rag_pipeline import RAGPipeline
        from config import settings as cfg

        set_runtime_config(
            st.session_state.get("provider", "gemini"),
            st.session_state.get("api_key", ""),
            st.session_state.get("model", "")
        )

        doc_id = str(uuid.uuid4())
        os.makedirs(cfg.upload_dir, exist_ok=True)
        tmp = os.path.join(cfg.upload_dir, f"{doc_id}.pdf")
        with open(tmp, "wb") as fp:
            fp.write(file_bytes)

        rag    = RAGPipeline(provider=st.session_state.get("provider", "gemini"))
        result = rag.ingest_document(tmp, "pdf", doc_id, filename)
        result["doc_id"] = doc_id
        return result, None
    except Exception as e:
        return None, str(e)


def direct_rag_call(func_name: str, **kwargs):
    """Call RAG pipeline directly without HTTP backend."""
    try:
        backend_path = _get_backend_path()
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.llm_factory import set_runtime_config
        from services.rag_pipeline import RAGPipeline

        set_runtime_config(
            st.session_state.get("provider", "gemini"),
            st.session_state.get("api_key", ""),
            st.session_state.get("model", "")
        )
        rag = RAGPipeline(provider=st.session_state.get("provider", "gemini"))
        fn  = getattr(rag, func_name)
        return fn(**kwargs), None
    except Exception as e:
        return None, str(e)


# ── TABS ──────────────────────────────────────────────────────────────
tabs = st.tabs(["📤 Upload", "💬 Q&A Chat", "📝 Summary", "🔍 Findings", "⚖️ Compare", "🔗 References"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Upload Research Papers")

    uploaded = st.file_uploader(
        "Choose one or more PDF research papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        for f in uploaded:
            already = any(d["name"] == f.name for d in st.session_state.r_docs)
            if already:
                st.info(f"Already uploaded: {f.name}")
                continue

            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 **{f.name}** ({f.size/1024:.1f} KB)")

            with col2:
                if st.button(f"Process", key=f"proc_{f.name}"):
                    if not check_key():
                        st.stop()

                    with st.spinner(f"Processing {f.name}..."):
                        file_bytes = f.getvalue()

                        # Try HTTP backend first
                        result, err = call_backend(
                            "/research/upload",
                            files={"file": (f.name, file_bytes, "application/pdf")}
                        )

                        # Fall back to direct Python call
                        if err:
                            if err != "backend_offline":
                                st.warning(f"HTTP error: {err}. Trying direct mode...")
                            result, err = direct_upload(file_bytes, f.name)

                        if err:
                            st.error(f"❌ {err}")
                        elif result:
                            doc_entry = {
                                "id":     result.get("doc_id", str(uuid.uuid4())),
                                "name":   f.name,
                                "pages":  result.get("page_count", "?"),
                                "chunks": result.get("chunk_count", "?"),
                            }
                            st.session_state.r_docs.append(doc_entry)
                            st.session_state.r_active_id   = doc_entry["id"]
                            st.session_state.r_active_name = doc_entry["name"]
                            st.session_state.r_chat        = []
                            st.success(f"✅ {f.name} ready! {doc_entry['pages']} pages · {doc_entry['chunks']} chunks")

    if st.session_state.r_docs:
        st.divider()
        st.markdown("### 📚 Uploaded Papers")
        for doc in st.session_state.r_docs:
            is_active = doc["id"] == st.session_state.r_active_id
            c1, c2, c3 = st.columns([5, 1, 1])
            with c1:
                prefix = "🟢 **[Active]**" if is_active else "⚪"
                st.markdown(f'<div class="doc-card">{prefix} {doc["name"]} — {doc["pages"]} pages · {doc["chunks"]} chunks</div>',
                            unsafe_allow_html=True)
            with c2:
                if not is_active and st.button("Select", key=f"sel_{doc['id']}"):
                    st.session_state.r_active_id   = doc["id"]
                    st.session_state.r_active_name = doc["name"]
                    st.session_state.r_chat        = []
                    st.rerun()
            with c3:
                if st.button("🗑️", key=f"del_{doc['id']}"):
                    st.session_state.r_docs = [d for d in st.session_state.r_docs if d["id"] != doc["id"]]
                    if st.session_state.r_active_id == doc["id"]:
                        st.session_state.r_active_id   = ""
                        st.session_state.r_active_name = ""
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Q&A CHAT
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("💬 Chat with Your Paper (RAG)")

    if not st.session_state.r_active_id:
        st.info("📤 Upload and select a paper in the **Upload** tab first.")
    else:
        st.markdown(f"📄 Active: **{st.session_state.r_active_name}**")

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.r_chat:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {msg["content"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bot">💡 <b>Assistant:</b><br>{msg["content"]}</div>',
                                unsafe_allow_html=True)
                    if msg.get("sources"):
                        with st.expander(f"📎 {len(msg['sources'])} source chunks used"):
                            for s in msg["sources"]:
                                st.markdown(
                                    f'<div class="source-box">📌 Chunk {s.get("chunk_index","?")} '
                                    f'| Score: {s.get("score","?")} | Page: {s.get("page","N/A")}<br>'
                                    f'<i>{s.get("preview","")}</i></div>',
                                    unsafe_allow_html=True
                                )

        st.markdown("**💡 Quick questions:**")
        quick_qs = [
            "What is the main objective?",
            "What methodology was used?",
            "What are the key results?",
            "What are the limitations?",
            "What future work is suggested?"
        ]
        cols = st.columns(5)
        for i, (col, q) in enumerate(zip(cols, quick_qs)):
            with col:
                if st.button(q, key=f"qq_{i}", use_container_width=True):
                    st.session_state["_pending_q"] = q

        question = st.chat_input("Ask anything about the paper...")
        if not question and "_pending_q" in st.session_state:
            question = st.session_state.pop("_pending_q")

        if question:
            if not check_key():
                st.stop()

            st.session_state.r_chat.append({"role": "user", "content": question})

            with st.spinner("🔍 Retrieving relevant passages and generating answer..."):
                result, err = call_backend("/research/ask", json={
                    "doc_id":       st.session_state.r_active_id,
                    "question":     question,
                    "chat_history": st.session_state.r_chat[:-1]
                })

                if err:
                    result, err = direct_rag_call(
                        "query",
                        question=question,
                        doc_id=st.session_state.r_active_id,
                        chat_history=st.session_state.r_chat[:-1]
                    )

                if err:
                    st.error(f"❌ {err}")
                elif result:
                    st.session_state.r_chat.append({
                        "role":    "assistant",
                        "content": result.get("answer", "No answer generated."),
                        "sources": result.get("sources", [])
                    })
                    st.rerun()

        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.r_chat = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — SUMMARY
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("📝 Paper Summarization")

    if not st.session_state.r_active_id:
        st.info("Select a paper in the Upload tab first.")
    else:
        st.markdown(f"📄 **{st.session_state.r_active_name}**")

        stype = st.radio(
            "Summary style:",
            ["structured", "brief", "beginner"],
            horizontal=True,
            format_func=lambda x: {
                "structured": "📊 Structured (Full academic breakdown)",
                "brief":      "⚡ Brief (3-5 sentences)",
                "beginner":   "🎓 Beginner-Friendly (Plain English)"
            }[x]
        )

        if st.button("✨ Generate Summary", type="primary"):
            if not check_key():
                st.stop()

            with st.spinner("Generating summary..."):
                result, err = call_backend("/research/summarize",
                                           json={"doc_id": st.session_state.r_active_id,
                                                 "summary_type": stype})
                if err:
                    result, err = direct_rag_call(
                        "summarize_document",
                        doc_id=st.session_state.r_active_id,
                        summary_type=stype
                    )

                if err:
                    st.error(f"❌ {err}")
                elif result:
                    st.session_state.r_last_summary = result.get("summary", "")
                    st.markdown("---")
                    st.markdown(result.get("summary", ""))
                    st.caption(f"Tokens used: {result.get('tokens_used', 'N/A')}")

        if st.session_state.r_last_summary:
            st.download_button("📥 Download Summary", st.session_state.r_last_summary,
                               file_name="summary.txt", mime="text/plain")


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — KEY FINDINGS
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🔍 Key Findings & Conclusions")

    if not st.session_state.r_active_id:
        st.info("Select a paper first.")
    else:
        st.markdown(f"📄 **{st.session_state.r_active_name}**")

        if st.button("🔍 Extract Key Findings", type="primary"):
            if not check_key():
                st.stop()

            with st.spinner("Analyzing paper for findings..."):
                result, err = call_backend("/research/key-findings",
                                           json={"doc_id": st.session_state.r_active_id})
                if err:
                    result, err = direct_rag_call(
                        "extract_key_findings",
                        doc_id=st.session_state.r_active_id
                    )

                if err:
                    st.error(f"❌ {err}")
                elif result:
                    st.session_state.r_last_findings = result.get("findings", "")
                    st.markdown(result.get("findings", ""))

        if st.session_state.r_last_findings:
            st.download_button("📥 Download Findings", st.session_state.r_last_findings,
                               file_name="key_findings.txt", mime="text/plain")


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARE
# ══════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("⚖️ Compare Multiple Papers")

    if len(st.session_state.r_docs) < 2:
        st.info("📤 Upload at least 2 papers to compare them.")
    else:
        docs = st.session_state.r_docs
        selected_ids = st.multiselect(
            "Select papers to compare (min 2):",
            options=[d["id"] for d in docs],
            format_func=lambda x: next(d["name"] for d in docs if d["id"] == x),
            default=[d["id"] for d in docs[:min(2, len(docs))]]
        )

        if len(selected_ids) >= 2:
            if st.button("⚖️ Compare Now", type="primary"):
                if not check_key():
                    st.stop()

                with st.spinner(f"Comparing {len(selected_ids)} papers..."):
                    names  = [next(d["name"] for d in docs if d["id"] == s) for s in selected_ids]
                    result, err = call_backend("/research/compare",
                                              json={"doc_ids": selected_ids, "doc_names": names})
                    if err:
                        result, err = direct_rag_call(
                            "compare_papers",
                            doc_ids=selected_ids,
                            doc_names=names
                        )

                    if err:
                        st.error(f"❌ {err}")
                    elif result:
                        st.session_state.r_comparison = result.get("comparison", "")
                        st.markdown(result.get("comparison", ""))

        if st.session_state.r_comparison:
            st.download_button("📥 Download Comparison", st.session_state.r_comparison,
                               file_name="paper_comparison.txt", mime="text/plain")


# ══════════════════════════════════════════════════════════════════════
# TAB 6 — REFERENCES
# ══════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🔗 References & Citations")

    if not st.session_state.r_active_id:
        st.info("Select a paper first.")
    else:
        st.markdown(f"📄 **{st.session_state.r_active_name}**")

        if st.button("📚 Extract References", type="primary"):
            if not check_key():
                st.stop()

            with st.spinner("Extracting references..."):
                result, err = call_backend("/research/references",
                                           json={"doc_id": st.session_state.r_active_id})
                if err:
                    result, err = direct_rag_call(
                        "extract_references",
                        doc_id=st.session_state.r_active_id
                    )

                if err:
                    st.error(f"❌ {err}")
                elif result:
                    st.session_state.r_last_refs = result.get("references", "")
                    st.markdown(result.get("references", ""))

        if st.session_state.r_last_refs:
            st.download_button("📥 Download References", st.session_state.r_last_refs,
                               file_name="references.txt", mime="text/plain")
