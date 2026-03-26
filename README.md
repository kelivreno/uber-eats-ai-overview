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
Embed query (Ollama)
    ↓
Top-K retrieval from Qdrant vector DB
    ↓
Retrieved context + query → LLM (Ollama)
    ↓
Natural language response

```



