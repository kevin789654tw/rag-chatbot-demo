<h1 align="center">🤖 rag-chatbot-demo</h1>

A lightweight RAG-based chatbot demo for fast prototyping and proof-of-concept experiments.

Built with LLMs, LangChain, FAISS, and Chainlit for rapid retrieval-augmented generation (RAG) testing.

---

## 📚 Table of Contents

- [📚 Table of Contents](#-table-of-contents)
- [🧠 Introduction](#-introduction)
- [🚀 Getting Started](#-getting-started)
  - [⚙️ Installation](#️-installation)
  - [🧩 Example Workflow](#-example-workflow)
  - [💬 Usage](#-usage)
- [✨ More details will be coming soon ✨](#-more-details-will-be-coming-soon-)

---

## 🧠 Introduction

**rag-chatbot-demo** is a minimal and flexible **Retrieval-Augmented Generation (RAG)** prototype.
It helps developers quickly validate RAG pipelines using **LLMs + Vector Search** without needing a complex backend.

This project is ideal for:

- 🧪 Rapid prototyping and experimentation
- ⚙️ Local proof-of-concept (POC) validation
- 📄 Testing document retrieval and contextual QA

Key technologies:

- **LLM Provider:** 🧠 OpenAI / 🤗 Hugging Face
- **Retriever:** 🧮 FAISS
- **Framework:** ⛓️ LangChain
- **Frontend:** 💬 Chainlit (chat interface)

<br/>

## 🚀 Getting Started

### ⚙️ Installation

Build from the source and install dependencies:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kevin789654tw/rag-chatbot-demo.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd rag-chatbot-demo
   ```

3. **Install dependencies:**

   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   ```bash
   cp .env.example .env
   ```

   ⚠️ Note: Replace placeholders with your own secrets, do not commit sensitive data.

### 🧩 Example Workflow

1. Prepare your document dataset (`.csv` or `Hugging Face dataset`)
2. Run `scripts/build_faiss_index.py` to generate FAISS index
3. Launch Chainlit interface
4. Ask questions and receive context-aware answers

Example:

```text
User: How many species of Eucalyptus are cultivated worldwide?
AI: Around 700 species of Eucalyptus are cultivated ...
```

### 💬 Usage

1. **Build FAISS Index:**
   Before running the chatbot, you need to build the index `faiss_index` for document retrieval:

   ```bash
   python -m scripts.build_faiss_index
   ```

2. **Run RAG Chatbot Demo:**
   Once the FAISS index is ready, you can start the Chainlit demo:

   ```bash
   python -m chainlit run app/demo/chainlit_app.py
   ```

   ⚠️ Note: Open the URL shown in the terminal (usually http://localhost:8000) to interact with the chatbot.

---

## ✨ More details will be coming soon ✨
