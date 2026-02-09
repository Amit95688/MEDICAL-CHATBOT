# Medical RAG - Simplified

> Simple healthcare document retrieval using RAG over PubMed documents. Fast, easy to understand, and easy to debug.

**Key Update:** Code simplified from 344 lines → 112 lines (66% reduction)

## Tech Stack

Python, Flask, LangChain, Sentence-Transformers, FAISS

---

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the app:**
   ```bash
   python app/main.py
   ```
   Watch the terminal output - every step is logged!

3. **Test it (in another terminal):**
   ```bash
   curl -X POST http://localhost:5000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is diabetes?"}'
   ```

---

## API

| Endpoint | Description |
|----------|-------------|
| **POST /api/query** | Ask a question, get relevant medical documents |
| **GET /api/health** | Health check |

---

## How It Works

1. Loads PubMed medical papers
2. Splits them into chunks
3. Creates embeddings (text → vectors)
4. Stores in FAISS index
5. When you ask a question → retrieves similar documents
6. Returns the document text

**No LLM. No complex chains. Just simple, fast RAG.**

---

## Config

Edit `app/config.py` to change:

- `DATASET_SAMPLE_SIZE` - How many papers to load (default: 200)
- `CHUNK_SIZE` - Document chunk size (default: 300)
- `RETRIEVER_K` - Top K documents to retrieve (default: 4)
- `EMBEDDING_MODEL_NAME` - Which model to use (default: sentence-transformers/all-MiniLM-L6-v2)
- `FAISS_INDEX_PATH` - Where to store index (default: data/faiss_index)

---

## Project Layout

```
app/
  config.py           # Simple settings
  main.py             # Flask app
  api/routes.py       # API endpoints
  rag/
    loader.py         # Load & chunk documents
    embeddings.py     # Create embeddings
    retriever.py      # FAISS indexing
    generator.py      # Retrieve & format
  templates/
    index.html        # Chat UI
data/
  faiss_index/        # Vector index (created automatically)
```

---


---

## Debug Tips

Watch the terminal output when running:

```
✓ Loading embeddings: sentence-transformers/all-MiniLM-L6-v2 on cpu
✓ Loading dataset: ccdv/pubmed-summarization, sample size: 200
✓ Created 1234 chunks
✓ Loading index from /path/to/faiss_index
✓ Retrieving top 4 documents
✓ Query: What is diabetes?
✓ Found 4 documents
```

---

## Common Issues

**"No documents found"**
- Index will be created automatically on first run
- Check `data/faiss_index/` directory

**"Module not found"**
- Run: `pip install -r requirements.txt`

**Import errors**
- Check terminal output - it says exactly what's wrong

---

## Next Steps

- Run the app: `python app/main.py`
- Try some queries
- Read the code - it's simple and understandable
- Add features as needed - the code is easy to modify
