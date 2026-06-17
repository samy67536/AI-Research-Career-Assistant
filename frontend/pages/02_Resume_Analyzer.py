# frontend/pages/02_Resume_Analyzer.py
import streamlit as st
import sys, os, uuid
import plotly.graph_objects as go
import plotly.express as px

# Fix sys.path so backend imports work in both HTTP and direct mode
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

st.set_page_config(page_title="Resume Analyzer", page_icon="📋", layout="wide")

st.markdown("""
<style>
.ats-high   { border-left: 5px solid #27AE60 !important; }
.ats-medium { border-left: 5px solid #F39C12 !important; }
.ats-low    { border-left: 5px solid #E74C3C !important; }
.skill-chip {
    display:inline-block; background:#E6F1FB; color:#1F3A6B;
    padding:3px 11px; border-radius:20px; font-size:0.82rem; margin:2px;
}
.missing-chip {
    display:inline-block; background:#FCEAEA; color:#C0392B;
    padding:3px 11px; border-radius:20px; font-size:0.82rem; margin:2px;
}
.suggest-item {
    background:#F0F9F0; border-left:4px solid #27AE60;
    padding:0.65rem 1rem; border-radius:4px; margin:0.35rem 0; font-size:0.92rem;
}
.cover-box {
    background:#F8F9FA; border:1px solid #DEE2E6;
    border-radius:8px; padding:1.2rem 1.5rem; font-size:0.93rem; line-height:1.7;
}
</style>
""", unsafe_allow_html=True)

st.title("📋 AI Resume Analyzer")
st.caption("ATS Score · Skill Extraction · JD Matching · AI Suggestions · Cover Letter")

for k, v in {"resume_result": None, "resume_text": "", "job_desc_cache": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def check_key():
    if not st.session_state.get("key_validated"):
        st.warning("⚠️ Set your API key on the **Home** page sidebar first.")
        return False
    return True


def _get_backend_path():
    return os.path.join(_root, "backend")


def run_analysis(file_bytes: bytes, filename: str, job_desc: str) -> dict:
    """Run resume analysis — direct Python call (no HTTP backend required)."""
    backend_path = _get_backend_path()
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from services.llm_factory import set_runtime_config
    from services.resume_analyzer import ResumeAnalyzer
    from services.document_processor import DocumentProcessor
    from config import settings as cfg

    set_runtime_config(
        st.session_state.get("provider", "gemini"),
        st.session_state.get("api_key", ""),
        st.session_state.get("model", "")
    )

    ext = "pdf" if filename.lower().endswith(".pdf") else "docx"
    doc_id   = str(uuid.uuid4())
    tmp_path = os.path.join(cfg.upload_dir, f"{doc_id}.{ext}")
    os.makedirs(cfg.upload_dir, exist_ok=True)

    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    try:
        proc = DocumentProcessor()
        text, meta = proc.extract_text(tmp_path, ext)
        st.session_state.resume_text     = text
        st.session_state.job_desc_cache  = job_desc

        analyzer = ResumeAnalyzer(provider=st.session_state.get("provider", "gemini"))
        result   = analyzer.analyze(text, job_desc)
        result["word_count"] = meta.get("word_count", 0)
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def ats_gauge(score: float):
    c = "#27AE60" if score >= 70 else "#F39C12" if score >= 50 else "#E74C3C"
    lbl = "Excellent ✅" if score >= 70 else "Good 🟡" if score >= 50 else "Needs Work 🔴"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 40, "color": c}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
            "bar":  {"color": c, "thickness": 0.28},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  50], "color": "#FCEAEA"},
                {"range": [50, 70], "color": "#FFF3CD"},
                {"range": [70,100], "color": "#EAF6EE"},
            ],
        },
        title={"text": f"ATS Compatibility — {lbl}", "font": {"size": 17}}
    ))
    fig.update_layout(height=290, margin=dict(t=60, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── LAYOUT ────────────────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.markdown("### 📤 Upload Resume")
    resume_file = st.file_uploader("PDF or DOCX", type=["pdf", "docx"])

    st.markdown("### 📋 Job Description")
    job_desc = st.text_area(
        "Paste JD for ATS matching (optional but recommended)",
        height=220,
        placeholder="We are looking for a Python Developer with 3+ years experience in ML...",
        value=st.session_state.job_desc_cache
    )

    analyze_btn = st.button(
        "🚀 Analyze Resume",
        type="primary",
        use_container_width=True,
        disabled=(resume_file is None)
    )

    if resume_file is None:
        st.caption("⬆️ Upload a resume to enable analysis")

    if analyze_btn and resume_file:
        if not check_key():
            st.stop()

        with st.spinner("📊 Extracting · Analyzing · Scoring..."):
            try:
                result = run_analysis(resume_file.getvalue(), resume_file.name, job_desc)
                st.session_state.resume_result = result
                st.success(f"✅ Analysis complete! ATS Score: **{result['ats_score']}/100**")
            except Exception as e:
                err_msg = str(e)
                st.error(f"❌ {err_msg}")
                if any(k in err_msg.lower() for k in ["credits", "quota", "balance", "billing"]):
                    st.info("💡 Switch to **Gemini** (free tier) on the Home page sidebar!")
                else:
                    st.info("💡 Make sure your API key is set correctly on the Home page.")

# ── RESULTS ───────────────────────────────────────────────────────────
with right:
    res = st.session_state.resume_result

    if res is None:
        st.info("⬅️ Upload a resume and click **Analyze Resume** to see results.")
        st.markdown("""
**What you'll get:**
- 🎯 ATS Score (0–100) with detailed breakdown
- 🧠 Technical & soft skill detection
- 📊 Resume vs Job Description matching
- ❌ Missing skills list
- ✅ 8 specific improvement suggestions
- ✉️ Personalized cover letter
        """)
    else:
        score = res["ats_score"]

        result_tabs = st.tabs([
            "🎯 ATS Score",
            "🧠 Skills",
            "📊 JD Match",
            "✅ Suggestions",
            "✉️ Cover Letter"
        ])

        with result_tabs[0]:
            col_gauge, col_breakdown = st.columns([1, 1])

            with col_gauge:
                st.plotly_chart(ats_gauge(score), use_container_width=True)
                s1, s2, s3 = st.columns(3)
                s1.metric("Words",      res.get("word_count", "?"))
                s2.metric("Tech Skills", len(res["skills_found"].get("technical", [])))
                s3.metric("Experience",  res.get("experience_years", "N/A"))

            with col_breakdown:
                st.markdown("#### 📊 Score Breakdown")
                bd = res.get("ats_breakdown", {})
                factor_labels = {
                    "keyword_match":              "🔤 Keyword Match",
                    "structure":                  "📐 Resume Structure",
                    "skills":                     "🧠 Skills Coverage",
                    "action_verbs":               "⚡ Action Verbs",
                    "education":                  "🎓 Education",
                    "quantified_achievements":    "📈 Quantified Results",
                }
                for key, label in factor_labels.items():
                    if key in bd:
                        d     = bd[key]
                        sv    = d.get("score", 0)
                        mx    = d.get("max", 10)
                        pct   = sv / mx if mx > 0 else 0
                        st.markdown(f"**{label}** — {sv}/{mx}")
                        st.progress(pct)

            st.divider()
            if score >= 70:
                st.success("🎉 Great ATS score! Your resume is well-optimized.")
            elif score >= 50:
                st.warning("⚠️ Decent score. Add more relevant keywords and quantify achievements.")
            else:
                st.error("🔴 Low ATS score. Review the Suggestions tab for improvements.")

        with result_tabs[1]:
            tech  = res["skills_found"].get("technical", [])
            soft  = res["skills_found"].get("soft", [])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### 💻 Technical Skills ({len(tech)})")
                if tech:
                    html = "".join([f'<span class="skill-chip">✓ {s}</span>' for s in tech])
                    st.markdown(html, unsafe_allow_html=True)
                    fig = go.Figure(go.Bar(
                        x=tech[:12], y=[1]*min(len(tech), 12),
                        marker_color="#2E6DAD",
                        text=tech[:12], textposition="auto"
                    ))
                    fig.update_layout(
                        height=220, showlegend=False,
                        yaxis=dict(showticklabels=False, showgrid=False),
                        xaxis=dict(showticklabels=False),
                        margin=dict(t=10, b=10, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No technical skills detected. Add a dedicated Skills section.")

            with c2:
                st.markdown(f"#### 🤝 Soft Skills ({len(soft)})")
                if soft:
                    html = "".join([f'<span class="skill-chip">✓ {s}</span>' for s in soft])
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.warning("No soft skills detected.")

                st.markdown("#### 🎓 Education")
                edu = res.get("education", [])
                if edu:
                    for e in edu:
                        st.markdown(f"• {e}")
                else:
                    st.info("Education details not clearly found.")

                st.markdown("#### ✅ Sections Found")
                sections = res.get("sections_found", {})
                for sec, found in sections.items():
                    icon = "✅" if found else "❌"
                    st.markdown(f"{icon} {sec.title()}")

        with result_tabs[2]:
            jd = res.get("jd_analysis", {})
            if not jd:
                st.info("📋 No job description was provided. Re-analyze with a JD for detailed matching.")
                if st.button("📋 Re-analyze with JD"):
                    st.session_state.resume_result = None
                    st.rerun()
            else:
                sim = jd.get("similarity_percent", 0)
                c1, c2, c3 = st.columns(3)
                c1.metric("JD Match", f"{sim}%",
                           delta="Strong match" if sim >= 60 else "Needs improvement")
                c2.metric("Skills Matched", len(jd.get("skills_matched", [])))
                c3.metric("Skills Missing", len(jd.get("skills_missing", [])))

                matched_n = len(jd.get("skills_matched", []))
                missing_n = len(jd.get("skills_missing", []))
                if matched_n + missing_n > 0:
                    fig = go.Figure(go.Pie(
                        labels=["Matched", "Missing"],
                        values=[matched_n, missing_n],
                        hole=0.5,
                        marker_colors=["#27AE60", "#E74C3C"]
                    ))
                    fig.update_layout(height=220, margin=dict(t=20,b=10,l=0,r=0),
                                      paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                col_m, col_x = st.columns(2)
                with col_m:
                    st.markdown("#### ✅ Matched Skills")
                    matched = jd.get("skills_matched", [])
                    if matched:
                        html = "".join([f'<span class="skill-chip">✓ {s}</span>' for s in matched])
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.warning("No matched skills found.")
                with col_x:
                    st.markdown("#### ❌ Missing Skills")
                    missing = jd.get("skills_missing", [])
                    if missing:
                        html = "".join([f'<span class="missing-chip">✗ {s}</span>' for s in missing])
                        st.markdown(html, unsafe_allow_html=True)
                        st.caption("Add these to your resume if you have them.")
                    else:
                        st.success("No critical skill gaps! 🎉")

        with result_tabs[3]:
            st.markdown("#### ✅ Improvement Suggestions")
            suggestions = res.get("suggestions", [])
            if suggestions:
                for i, s in enumerate(suggestions, 1):
                    st.markdown(f'<div class="suggest-item"><b>{i}.</b> {s}</div>',
                                unsafe_allow_html=True)
            else:
                st.info("No suggestions generated. Try re-analyzing with a job description.")

            st.divider()
            st.markdown("#### ✏️ AI-Generated Optimized Summary")
            if st.button("🔄 Generate Optimized Professional Summary"):
                if not check_key():
                    st.stop()
                with st.spinner("Writing optimized summary..."):
                    try:
                        backend_path = _get_backend_path()
                        if backend_path not in sys.path:
                            sys.path.insert(0, backend_path)
                        from services.llm_factory import set_runtime_config
                        from services.resume_analyzer import ResumeAnalyzer
                        set_runtime_config(
                            st.session_state.get("provider", "gemini"),
                            st.session_state.get("api_key", ""),
                            st.session_state.get("model", "")
                        )
                        svc = ResumeAnalyzer(provider=st.session_state.get("provider", "gemini"))
                        opt = svc.generate_optimized_summary(
                            st.session_state.resume_text,
                            st.session_state.job_desc_cache
                        )
                        st.markdown(f'<div class="cover-box">{opt}</div>', unsafe_allow_html=True)
                        st.download_button("📥 Download Summary", opt,
                                           file_name="optimized_summary.txt", mime="text/plain")
                    except Exception as e:
                        st.error(str(e))

        with result_tabs[4]:
            st.markdown("#### ✉️ AI-Generated Cover Letter")
            cover = res.get("cover_letter", "")
            if cover:
                st.markdown(f'<div class="cover-box">{cover.replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True)
                st.divider()
                col_dl, col_copy = st.columns(2)
                with col_dl:
                    st.download_button(
                        "📥 Download Cover Letter (.txt)",
                        cover,
                        file_name="cover_letter.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col_copy:
                    st.text_area("📋 Copy from here:", cover, height=100, key="cover_copy")
            else:
                st.info("📋 Provide a job description and re-analyze to generate a personalized cover letter.")
                if st.session_state.resume_text and st.button("✉️ Generate Cover Letter Now"):
                    if not check_key():
                        st.stop()
                    jd_input = st.text_area("Enter Job Description:", height=150, key="jd_for_cover")
                    if jd_input:
                        with st.spinner("Writing cover letter..."):
                            try:
                                backend_path = _get_backend_path()
                                if backend_path not in sys.path:
                                    sys.path.insert(0, backend_path)
                                from services.llm_factory import set_runtime_config
                                from services.resume_analyzer import ResumeAnalyzer
                                set_runtime_config(
                                    st.session_state.get("provider", "gemini"),
                                    st.session_state.get("api_key", ""),
                                    st.session_state.get("model", "")
                                )
                                svc   = ResumeAnalyzer(provider=st.session_state.get("provider", "gemini"))
                                cover = svc._generate_cover_letter(st.session_state.resume_text, jd_input)
                                st.markdown(f'<div class="cover-box">{cover}</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(str(e))
