# RAG Chatbot: talk to your documents using Deepseek-R1 model

This repository demonstrates how to build a **Retrieval-Augmented Generation (RAG)** chatbot using:

- **[Ollama](https://github.com/jmorganca/ollama)** (a local LLM runner)
- **[LlamaIndex](https://github.com/jerryjliu/llama_index)** (for document ingestion, vector indexing, retrieval, and a simple chat interface)
- **[Streamlit](https://streamlit.io/)** (for building a user-friendly web UI)
- **Hugging Face Embeddings** (for creating vector embeddings from text)

Users can **upload PDF/CSV documents** or a **pre-vectorized file**, then **ask questions** about their content. The chatbot retrieves context from your docs before calling Ollama to generate its final answer.

---

## Features

- **Document Ingestion**: Upload PDFs (parsed via `SimpleDirectoryReader`) or CSVs (transformed into text rows).
- **Index & Retrieval**: Creates a **VectorStoreIndex** using **HuggingFace** embeddings for semantic similarity.
- **Local Ollama**: Utilizes your local Ollama server to run large language models on macOS.
- **Caching**: Saves the built index as a `.pkl` file so you can reload it without re-indexing.
- **Interactive Streamlit UI**: Chat messages appear in a neat conversation-style interface, with a text input to query.

![](Demo.gif)
---
![](Demo.gif)

## Prerequisites

1. **Ollama**
2. **GPU** is optional; you can run CPU-only or leverage a CUDA GPU for embeddings.

## Installation

### 1. Clone or download this repository
```bash
git clone https://github.com/aditya1000/Rag-Deepseek-Chatbot.git
cd Rag-Deepseek-Chatbot
```


### (Optional) Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

#### macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

#### Windows:
``` bash
python -m venv venv
.\venv\Scripts\activate
```

#### Install Dependencies
Using the provided requirements.txt:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Or install the required packages manually:
```bash
pip install llama_index streamlit torch pandas llama_index.embeddings.huggingface numpy==1.26.4
```


### Verify Ollama Installation
Ensure that Ollama is properly installed:

```bash
which ollama  # macOS/Linux: should show the path to the Ollama binary
# or
where ollama  # Windows: should show the path to Ollama binary
```

#### Then, pull the required model:
```bash
ollama pull deepseek-r1:7b
```

If you see an error, double-check your Homebrew installation and PATH settings.

# Usage

## Start the Chatbot
```bash
streamlit run chatbot.py
```

Open in Browser

Streamlit will display the app's local URL, usually:

http://localhost:8501
