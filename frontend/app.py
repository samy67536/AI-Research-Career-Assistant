# frontend/app.py
import streamlit as st
import requests
import os
import sys

# Ensure backend is importable when running in direct mode
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

st.set_page_config(
    page_title="Research & Career Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1F3A6B 0%, #2E6DAD 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;
}
.main-header h1 { color: white; font-size: 2rem; margin: 0; }
.main-header p  { color: #BDD7EE; margin: 0.4rem 0 0; }
.key-box {
    background: #F0F4F8; border: 1.5px solid #2E6DAD;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.success-box {
    background: #EAF6EE; border-left: 4px solid #27AE60;
    border-radius: 6px; padding: 0.7rem 1rem; margin: 0.4rem 0;
}
.error-box {
    background: #FCEAEA; border-left: 4px solid #E74C3C;
    border-radius: 6px; padding: 0.7rem 1rem; margin: 0.4rem 0;
}
.step-box {
    background: #F8F9FA; border-radius: 8px; padding: 1rem 1.2rem;
    border-left: 4px solid #2E6DAD; margin: 0.5rem 0;
}
.free-badge {
    background: #27AE60; color: white; font-size: 0.72rem; font-weight: 700;
    padding: 2px 7px; border-radius: 10px; margin-left: 6px; vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

BACKEND = "http://localhost:8000"

# ── Session state defaults ────────────────────────────────────────────
for k, v in {
    "api_key": "", "provider": "gemini", "model": "",
    "key_validated": False, "key_status_msg": ""
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # ── Provider selection ─────────────────────────────────────────
    st.markdown("### Step 1: Choose LLM Provider")
    provider = st.selectbox(
        "Provider",
        ["gemini", "openai", "claude", "deepseek"],
        index=["gemini", "openai", "claude", "deepseek"].index(st.session_state.provider),
        format_func=lambda x: {
            "claude":   "🟣 Claude (Anthropic) — paid",
            "openai":   "🟢 GPT (OpenAI) — paid",
            "gemini":   "🔵 Gemini (Google) — FREE tier ✅",
            "deepseek": "🟠 DeepSeek — very cheap",
        }[x]
    )
    st.session_state.provider = provider

    if provider == "gemini":
        st.info("💡 **Gemini has a FREE tier** — no credit card needed!\nGet your key in seconds at aistudio.google.com")

    # Model selector
    model_options = {
        "claude":   [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        "openai":   ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "gemini":   ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        "deepseek": ["deepseek-chat", "deepseek-coder"],
    }
    model = st.selectbox("Model", model_options[provider])
    st.session_state.model = model

    st.divider()

    # ── API Key input ──────────────────────────────────────────────
    st.markdown("### 🔑 Step 2: Enter API Key")

    key_links = {
        "claude":   "https://console.anthropic.com",
        "openai":   "https://platform.openai.com/api-keys",
        "gemini":   "https://aistudio.google.com/apikey",
        "deepseek": "https://platform.deepseek.com",
    }
    key_placeholders = {
        "claude":   "sk-ant-api03-...",
        "openai":   "sk-proj-...",
        "gemini":   "AIzaSy...",
        "deepseek": "sk-...",
    }

    st.caption(f"🔗 Get key: [{key_links[provider]}]({key_links[provider]})")

    api_key = st.text_input(
        f"{provider.upper()} API Key",
        type="password",
        placeholder=key_placeholders[provider],
        value=st.session_state.api_key,
        help="Your key is only stored in session memory — never saved to disk."
    )
    st.session_state.api_key = api_key

    # ── Validate & Apply button ────────────────────────────────────
    st.divider()
    st.markdown("### ✅ Step 3: Validate & Apply")

    col_val, col_apply = st.columns(2)

    with col_val:
        if st.button("🔍 Test Key", use_container_width=True):
            if not api_key or len(api_key) < 10:
                st.error("Enter a valid API key first.")
            else:
                with st.spinner("Testing..."):
                    # Try backend first, fall back to direct call
                    tested = False
                    try:
                        resp = requests.post(f"{BACKEND}/settings/validate-key",
                                             json={"provider": provider,
                                                   "api_key": api_key,
                                                   "model": model},
                                             timeout=30)
                        if resp.status_code == 200:
                            st.session_state.key_validated = True
                            st.success("✅ Key is working!")
                        else:
                            detail = resp.json().get("detail", resp.text)
                            st.error(f"❌ {detail}")
                        tested = True
                    except requests.exceptions.ConnectionError:
                        pass

                    if not tested:
                        # Direct validation fallback
                        try:
                            from backend.services.llm_factory import validate_api_key, set_runtime_config
                            set_runtime_config(provider, api_key, model)
                            result = validate_api_key(provider, api_key)
                            if result["valid"]:
                                st.session_state.key_validated = True
                                st.success("✅ Key is working!")
                            else:
                                st.error(result["message"])
                        except Exception as e:
                            st.error(str(e))

    with col_apply:
        if st.button("💾 Apply", use_container_width=True, type="primary"):
            if not api_key or len(api_key) < 10:
                st.error("Enter API key first.")
            else:
                with st.spinner("Applying..."):
                    try:
                        resp = requests.post(f"{BACKEND}/settings/configure",
                                             json={"provider": provider,
                                                   "api_key": api_key,
                                                   "model": model},
                                             timeout=15)
                        if resp.status_code == 200:
                            st.session_state.key_validated = True
                            st.success(f"✅ {provider.upper()} active!")
                        else:
                            st.error(resp.json().get("detail", "Failed"))
                    except requests.exceptions.ConnectionError:
                        # Backend not running — store in session for direct use
                        try:
                            from backend.services.llm_factory import set_runtime_config
                            set_runtime_config(provider, api_key, model)
                            st.session_state.key_validated = True
                            st.success(f"✅ {provider.upper()} key saved (direct mode)!")
                        except Exception as e:
                            st.error(str(e))

    # Status indicator
    if st.session_state.key_validated:
        st.markdown('<div class="success-box">🟢 <b>Ready</b> — AI features active</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">🔴 <b>Not configured</b> — Enter & apply key</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.caption("📄 **Pages:**\n- Research Assistant\n- Resume Analyzer")
    st.caption("💡 Key stored in session only — never saved to disk.")


# ── MAIN PAGE ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Research &amp; Career Assistant</h1>
    <p>Platform for Research Paper Analysis &amp; Resume Evaluation</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.key_validated:
    st.warning("⚠️ **Setup required** — Configure your API key in the sidebar to get started.")
    st.markdown("---")
    st.markdown("## 🚀 Quick Setup Guide")

    steps = [
        ("1️⃣ Choose Provider", "**Gemini is recommended** — it has a FREE tier with no credit card needed."),
        ("2️⃣ Get API Key", "Click the link shown in the sidebar to get your key (takes ~30 seconds)."),
        ("3️⃣ Enter Key", "Paste your API key in the password field in the sidebar."),
        ("4️⃣ Apply", "Click **Apply** to activate. Use **Test Key** to verify it works first."),
        ("5️⃣ Use the App", "Navigate to Research Assistant or Resume Analyzer using the sidebar."),
    ]
    for title, desc in steps:
        st.markdown(f'<div class="step-box"><b>{title}</b><br>{desc}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔑 Where to Get API Keys")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**🔵 Gemini** ⭐\n\naistudio.google.com\n\n🆓 **FREE tier — start here!**")
    with col2:
        st.markdown("**🟢 OpenAI**\n\nplatform.openai.com\n\n💳 Paid (add credits to billing)")
    with col3:
        st.markdown("**🟣 Claude**\n\nconsole.anthropic.com\n\n💳 Paid (add credits)")
    with col4:
        st.markdown("**🟠 DeepSeek**\n\nplatform.deepseek.com\n\n💰 Very cheap pricing")

    st.info("💡 **If OpenAI or Claude say 'insufficient balance'**: add credits on their billing page, OR switch to **Gemini** (free).")

else:
    st.success(f"✅ **{st.session_state.provider.upper()} ({st.session_state.model})** is active and ready!")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### 📄 Research Paper Assistant
- 📤 Upload PDF research papers
- 💬 **Context-aware Q&A** — ask anything about the paper
- 📝 Structured / Brief / Beginner summaries
- 🔍 Key findings & conclusions extraction
- ⚖️ Compare multiple papers side-by-side
- 🔗 Reference & citation extraction
        """)
        st.info("👉 Open **Research Assistant** in the sidebar")

    with col2:
        st.markdown("""
### 📋 Resume Analyzer
- 📤 Upload PDF or DOCX resumes
- 🎯 **ATS Score** (0-100) with breakdown
- 🧠 Technical & soft skill extraction
- 📊 Resume vs Job Description matching
- ❌ Missing skills identification
- ✅ 8 AI improvement suggestions
- ✉️ Personalized cover letter generation
        """)
        st.info("👉 Open **Resume Analyzer** in the sidebar")

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Provider",  st.session_state.provider.upper())
    c2.metric("Model",     st.session_state.model or "Default")
    c3.metric("RAG Engine","FAISS + MiniLM")
    c4.metric("File Types","PDF + DOCX")
