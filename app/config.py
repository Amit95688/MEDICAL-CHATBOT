"""Simple config for Medical RAG app."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# PATHS
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index")

# DATA
DATASET_NAME = "ccdv/pubmed-summarization"
DATASET_SPLIT = "train"
DATASET_SAMPLE_SIZE = 200

# CHUNKING
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# EMBEDDINGS
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# RETRIEVAL
RETRIEVER_K = 4
