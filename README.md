# UberEats AI Overview
A natural language search system over 100,000+ US Uber Eats restaurants. 
Ask questions like "cheap sushi in LA" or "best pizza under $15 in Chicago" 
and get answers grounded in real menu and restaurant data.

Built as a learning project on RAG systems.

Covers: restaurant names, categories, price ranges, locations, 
and menu items across the US.

Flow:
```
Query
    ↓
Embed query (nomic-embed text)
    ↓
Top-K retrieval from Qdrant vector DB
    ↓
Retrieved context + query → LLM (Ollama)
    ↓
Natural language response

```

## Features

- Semantic search using embeddings
- Top-K retrieval from a vector database
- Natural language responses using a local LLM
- Streamlit interface for querying
- Supports restaurant metadata such as name, category, price range, and location

## Tech Stack

- Embeddings: `nomic-embed-text` via Ollama
- Vector database: Qdrant
- LLM: Ollama (needs to download the 3.2)
- Framework: Streamlit
- Orchestration: `llama_index`

## Setup

### 1. Clone repository

```bash
git clone https://github.com/kelivreno/uber-eats-ai-overview.git
cd uber-eats-ai-overview
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Qdrant

```bash
docker run -d \
  --name qdrantRagDB \
  -p 6333:6333 \
  -v "./qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

### 4. Install Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### 5. Run the application

```bash
streamlit run ubereats_streamlit_app.py
```

## Example Query

```text
cheap sushi in LA
```

The system returns:
- Retrieved restaurant entries
- Metadata such as price range, category, and location
- An LLM-generated answer based on retrieved context

## Limitations

- Retrieval quality depends on embedding quality
- No geographic distance filtering
- Ranking is primarily based on vector similarity
- Review integration still needs improvement
- Requires local setup with Ollama and Qdrant

## Purpose

This project is focused on learning:
- How embeddings represent text as vectors
- How vector databases enable semantic retrieval
- How RAG systems combine retrieval and generation
- How to build an end-to-end local AI application
