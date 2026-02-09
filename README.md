# MEDICAL-CHATBOT

> GenAI-driven healthcare document intelligence: RAG over PubMed-style documents with LangChain, FAISS, and optional LLM. Optimized for **speed** and **simplicity** (fast path by default; full RAG optional).

## Tools & Technologies

Python, Flask, LangChain, Sentence-Transformers, FAISS, Phi-2 (optional), React-friendly API

---

## Quick start (fast predictions)

1. **Build the index once** (so the app doesn’t build on first request):

   ```bash
   cd /path/to/GEN-AI
   python -m scripts.build_index
   ```

2. **Start the app:**

   ```bash
   python -m app.main
   ```

3. Open **http://localhost:5000/** and ask a question. Answers use **retrieved context only** (no LLM), so they’re fast.

---

## API (React or any client)

| Endpoint | Description |
|----------|-------------|
| **GET /** | HTML chat UI |
| **GET /api/health** | Health check |
| **POST /api/query** | **Fast:** retrieve + format (no LLM). Body: `{"question": "..."}` |
| **POST /api/query_full** | **Full RAG:** retrieve + LLM (slower). Body: `{"question": "..."}` |
| **POST /api/summarize** | Short summary of retrieved chunks. Body: `{"question": "..."}` |

---

## Why it was slow and how it’s fixed

- **Before:** First request built the FAISS index and loaded Phi-2; predictions took ~3 minutes and sometimes hung. Phi-2 with `device_map="auto"` caused **“Tensor on device meta is not on the expected device cuda:0”**.
- **Now:**
  1. **Fast path (default):** `/api/query` only retrieves and formats chunks (no LLM). Quick after the index is loaded.
  2. **Prebuild:** Run `python -m scripts.build_index` once; the app loads the index from disk instead of building on first request.
  3. **Device fix:** LLM (used only for `/api/query_full`) loads with an explicit device (`LLM_DEVICE=cpu` by default), no `device_map="auto"`, so the meta/cuda error is avoided.
  4. **Optional full RAG:** Use `POST /api/query_full` when you want an LLM-generated answer; keep `/api/query` for fast answers.

---

## Config (env)

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FAST_PATH` | `1` | Use fast path (retrieve only) for `query_rag()` when true |
| `FAISS_INDEX_PATH` | `app/data/faiss_index` | Where the FAISS index is stored |
| `EMBEDDING_DEVICE` | `cpu` | Device for embeddings |
| `LLM_DEVICE` | `cpu` | Device for LLM (avoids meta/cuda error; use `cuda` if you have GPU) |
| `RETRIEVER_K` | `4` | Number of chunks to retrieve |
| `DATASET_SAMPLE_SIZE` | `200` | Number of PubMed articles to index |

---

## Project layout (for learning)

```
app/
  config.py       # All settings (paths, data, embeddings, retriever, LLM)
  main.py          # Flask app and / route
  api/routes.py    # /api/health, /api/query, /api/query_full, /api/summarize
  rag/
    loader.py      # Load PubMed dataset and chunk
    embeddings.py  # HuggingFace embeddings
    retriever.py   # FAISS index load/build and retriever
    generator.py   # Fast answer (retrieve only) + full RAG (optional LLM)
  templates/
    index.html     # Chat UI
scripts/
  build_index.py   # Build FAISS index once
```

---

## Scale / enhance later

- Switch to Qdrant or another vector DB by replacing `app/rag/retriever.py` with a client that uses the same `get_embeddings()` and document format.
- Enable GPU: set `EMBEDDING_DEVICE=cuda` and `LLM_DEVICE=cuda` (and ensure the LLM loads with the fixed device logic in `generator.py`).
- Add auth, rate limits, and deployment config (e.g. Docker/AWS) as needed.
