# Research & Career Assistant
### Platform · Research Paper Analysis · Resume Evaluation

---

## ⚡ Quick Start (3 Steps)

### Step 1 — Install
```bash
cd ai-research-career-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Step 2 — Run
```bash
# Terminal 1 — Backend API
cd backend
python main.py
# → http://localhost:8000/docs

# Terminal 2 — Frontend UI
streamlit run frontend/app.py
# → http://localhost:8501
```

### Step 3 — Configure API Key in the UI
1. Open http://localhost:8501
2. In the **sidebar**: choose your LLM provider
3. Get your API key (links below) and paste it
4. Click **Test Key** to verify → then **Apply**
5. Go to Research Assistant or Resume Analyzer

---

## 🔑 API Keys — Free Tiers Available

| Provider  | Get API Key                           | Free Tier  |
|-----------|---------------------------------------|------------|
| 🟣 Claude  | https://console.anthropic.com         | $5 credit  |
| 🟢 OpenAI  | https://platform.openai.com/api-keys  | $5 credit  |
| 🔵 Gemini  | https://aistudio.google.com/apikey    | Free tier  |
| 🟠 DeepSeek| https://platform.deepseek.com         | Very cheap |

> **No .env editing needed** — enter your key directly in the app sidebar!

---

## 🔧 Troubleshooting 401 / Auth Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `invalid x-api-key` | Wrong/placeholder key in .env | Enter real key in sidebar → Apply |
| `authentication_error` | Key expired or wrong provider | Get a new key from the provider dashboard |
| `Connection refused` | Backend not running | Run `cd backend && python main.py` |
| `module not found` | Dependencies not installed | Run `pip install -r requirements.txt` |

---

## 📁 Project Structure
```
ai-research-career-assistant/
├── .env                          ← API keys (optional — can use UI instead)
├── requirements.txt
├── backend/
│   ├── main.py                   ← FastAPI app (all REST endpoints)
│   ├── config.py                 ← Settings loader
│   ├── database/connection.py    ← SQLAlchemy models
│   └── services/
│       ├── llm_factory.py        ← Claude/OpenAI/Gemini/DeepSeek factory
│       ├── document_processor.py ← PDF + DOCX extraction & chunking
│       ├── vector_store.py       ← FAISS embedding & similarity search
│       ├── rag_pipeline.py       ← Full RAG orchestration
│       └── resume_analyzer.py   ← ATS scoring & AI analysis
├── frontend/
│   ├── app.py                    ← Home + API key configuration UI
│   └── pages/
│       ├── 01_Research_Assistant.py
│       └── 02_Resume_Analyzer.py
└── data/
    ├── uploads/                  ← Uploaded files (auto-created)
    └── vectors/                  ← FAISS indexes (auto-created)
```

---

## Retrieval Pipeline Flow
```
PDF Upload
    ↓
Text Extraction (PyPDF + PDFPlumber)
    ↓
Chunking (1000 chars, 200 overlap)
    ↓
Embedding (all-MiniLM-L6-v2, 384-dim)
    ↓
FAISS Index (cosine similarity)
    ↓
User Question → Query Embedding
    ↓
Top-5 Chunk Retrieval
    ↓
RAG Prompt (system + context + question)
    ↓
LLM Response (Claude / GPT / Gemini)
    ↓
Answer + Source Citations
```

---

## 🐳 Docker (Optional)
```bash
docker-compose up --build
# Frontend → http://localhost:8501
# Backend  → http://localhost:8000
```
