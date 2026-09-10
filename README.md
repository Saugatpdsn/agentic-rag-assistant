# 🧠 Agentic RAG Assistant

An AI assistant that can answer questions from my documents and also search the web when needed.

I built this project to learn and practice **RAG, AI Agents, LangChain, LangGraph, tool calling, and LLM applications**.

The frontend is built with React and the backend uses FastAPI.

---

## ✨ Features

- 🤖 AI agent using LangGraph
- 📚 Ask questions from uploaded documents
- 🌐 Search the web when needed
- ⚡ Streaming responses
- 🔗 Show sources used for the answer
- 🧠 Conversation memory
- 📄 Upload PDF, TXT, and Markdown files
- 🎨 React + Tailwind CSS interface
- 🔍 LangSmith for tracing and debugging

---

## 🏗️ How It Works

The basic workflow is:

```text
User Question
      ↓
   AI Agent
      ↓
  ┌───┴────┐
  ↓        ↓
Documents  Web Search
  ↓        ↓
  └───┬────┘
      ↓
  AI Agent
      ↓
Final Answer