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
