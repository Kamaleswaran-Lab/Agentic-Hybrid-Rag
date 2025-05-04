# Open-Source Agentic Hybrid RAG Pipeline

> An end-to-end, framework for automated scientific literature review using an agentic hybrid RAG approach.

![Pipeline Overview](assets/general_workflow.png)  
*Figure 1: Pipeline Architecture.*  

![KG](assets/kg.png) 
*Figure 2: Knowledge Graph Schematic Model.*

![Cypher](assets/cypher.png) 
*Figure 3: KG Retrieval Function with Cypher query mode.*


![VS](assets/sim.png) 
*Figure 3: VS Retrieval Function.*

## Table of Contents
- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
  - [Using Docker](#using-docker)  
  - [Using Python Virtualenv](#using-python-virtualenv)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Directory Structure](#directory-structure)  
- [License](#license)  

## Features
- Ingest bibliometric data (PubMed, ArXiv, Google Scholar)  
- Build Neo4j knowledge graph with citation relationships  
- Chunk & embed full-text PDFs into FAISS vector store using LLaMA-3  
- Agentic orchestration:  
  - **GraphRAG** (Cypher queries over KG)  
  - **VectorRAG** (BM25 + dense retrieval + reranking)  
- Instruction tuning with Mistral-7B-Instruct-v0.2   
- Bootstrapped evaluation with error estimates  

## Prerequisites
- Docker Engine ≥ 20.10  
- Python ≥ 3.11 (if running natively)  
- Neo4j Community Edition ≥ 4.4  
- FAISS-compatible hardware  

## Installation

### Using Docker
1. Clone the repository:  
  ```bash
    git clone https://github.com/Kamaleswaran-Lab/Agentic-Hybrid-Rag.git
```

2. Build the Docker image:
  ```bash
    docker build -t agentic-rag:latest .
  ```

3. Run the container (adjust ports & volumes as needed):

```bash
  docker run -d \
    --name agentic-rag \
    -p 7474:7474 \          # Neo4j Browser
    -p 7687:7687 \          # Neo4j Bolt
    -p 5000:5000 \          # Agent HTTP API
    -v $(pwd)/data:/app/data \
    agentic-rag:latest
```

Access:
Neo4j Browser: http://localhost:7474
Agent API / UI: http://localhost:5000

### Using Python Virtualenv
Clone the repository and enter directory.
Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Ensure Neo4j is running locally or remotely, and update config.yml accordingly.
