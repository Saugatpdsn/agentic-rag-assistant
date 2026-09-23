<div align="center">

# 🧠 Agentic RAG Assistant

### An AI assistant that can chat with your documents and search the web when needed.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

[![LangChain](https://img.shields.io/badge/LangChain-1.3-1C3C3C?style=for-the-badge)](https://python.langchain.com/)

[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1C3C3C?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

[![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)

</div>

---

## 📖 About The Project

**Agentic RAG Assistant** is an AI chatbot that can answer questions using information from uploaded documents and the web.

I built this project to understand how **Agentic RAG systems** work in practice.

The project combines:

- **RAG** for searching information from documents
- **LangChain** for working with LLMs and tools
- **LangGraph** for building the AI agent
- **Google Gemini** for chat + embeddings
- **ChromaDB** for storing document embeddings
- **Web search** for finding additional information
- **Streamlit** as the single UI + application layer (no separate backend)
- **LangSmith** for tracing and debugging

The main idea is that the AI does not always have to use the same process. It can decide whether it should search the uploaded documents, search the web, or directly answer the question.

---

## ✨ Features

### 🤖 AI Agent

The application uses a LangGraph-based agent that can decide which tool it needs.

For example:

```text
User Question
     ↓
   Agent
     ↓
  Decide
  ├── Search Documents
  ├── Search Web
  └── Answer
```

---

## 🚀 Running locally

Everything — the LangGraph agent and the UI — runs in a single Streamlit process. There's no separate server to start.

**1. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your API key**

Copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY` (get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)) — and optionally `TAVILY_API_KEY` (without it, web search falls back to keyless DuckDuckGo).

By default this app uses `gemini-2.5-flash-lite` and `gemini-2.5-flash` — both covered by the Gemini API's free tier (Google AI Studio) and stable until Gemini 2.5's shutdown on **16 Oct 2026**. If you hit a `429` rate-limit error, you're past the free tier's requests-per-minute/day cap for that model; wait a bit or switch `MODEL_FAST`/`MODEL_HEAVY` in `.env` to another free-tier model.

> **About the `400 missing thought_signature` error:** Gemini's thinking models attach a `thought_signature` to function-call parts and require it echoed back on every step of a tool-calling turn. `langchain-google-genai` doesn't yet do this reliably through LangGraph's tool loop ([tracked upstream bug](https://github.com/langchain-ai/langchain-google/issues/1364)), on both Gemini 2.5 and 3.x. This app sets `thinking_budget=0` in `llms.py` to disable thinking entirely, so no signature is ever generated — that's what actually avoids the error, not the model choice itself.

**3. Run the app**

```bash
streamlit run frontend_streamlit/app.py
```

Run this from the project root (not from inside `frontend_streamlit/`) so the relative paths in `config.py` — `.env`, `./chroma_db`, `./data/sample_docs` — resolve correctly.

Then open `http://localhost:8501`.

On first launch the app auto-ingests the sample docs in `data/sample_docs/` if the vector store is empty. Upload more files any time from the sidebar.
