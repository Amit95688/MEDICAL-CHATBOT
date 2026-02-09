"""
Configuration for the Medical RAG app.

Organized for learning: paths, data, embeddings, retriever, and optional LLM.
Override with environment variables (e.g. USE_FAST_PATH=0, LLM_DEVICE=cuda).
"""
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", str(DATA_DIR / "faiss_index"))

# -----------------------------------------------------------------------------
# Data (PubMed summarization)
# -----------------------------------------------------------------------------
DATASET_NAME = "ccdv/pubmed-summarization"
DATASET_SPLIT = "train"
DATASET_SAMPLE_SIZE = int(os.environ.get("DATASET_SAMPLE_SIZE", "200"))

# Chunking for RAG
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# -----------------------------------------------------------------------------
# Embeddings (sentence-transformers)
# -----------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")

# -----------------------------------------------------------------------------
# Retriever (FAISS similarity search)
# -----------------------------------------------------------------------------
RETRIEVER_SEARCH_TYPE = "similarity"
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", "4"))  # more chunks = richer fast answer

# -----------------------------------------------------------------------------
# Fast path vs full RAG
# -----------------------------------------------------------------------------
# Fast path = retrieve + format only (no LLM). Quick and no GPU/device issues.
USE_FAST_PATH = os.environ.get("USE_FAST_PATH", "1").strip().lower() in ("1", "true", "yes")

# -----------------------------------------------------------------------------
# LLM (only used when USE_FAST_PATH is False or when calling /api/query_full)
# -----------------------------------------------------------------------------
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "microsoft/phi-2")
# Use "cpu" to avoid "Tensor on device meta is not on the expected device cuda:0"
LLM_DEVICE = os.environ.get("LLM_DEVICE", "cpu")
LLM_MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "128"))
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.95
