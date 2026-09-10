<div align="center">

# 🧠 Agentic RAG Assistant

### An AI assistant that can chat with your documents and search the web when needed.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

[![LangChain](https://img.shields.io/badge/LangChain-1.3-1C3C3C?style=for-the-badge)](https://python.langchain.com/)

[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1C3C3C?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)

[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)

[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)

[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)

</div>

---

## 📖 About The Project

**Agentic RAG Assistant** is an AI chatbot that can answer questions using information from uploaded documents and the web.

I built this project to understand how **Agentic RAG systems** work in practice.

The project combines:

- **RAG** for searching information from documents
- **LangChain** for working with LLMs and tools
- **LangGraph** for building the AI agent
- **ChromaDB** for storing document embeddings
- **Web search** for finding additional information
- **FastAPI** for the backend
- **React + TypeScript** for the frontend
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