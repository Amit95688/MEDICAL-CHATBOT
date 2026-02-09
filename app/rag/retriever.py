"""
FAISS vector store and retriever.

- Load index from disk if it exists (fast).
- Build once with: python -m scripts.build_index
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.config import FAISS_INDEX_PATH, RETRIEVER_SEARCH_TYPE, RETRIEVER_K
from app.rag.embeddings import get_embeddings
from app.rag.loader import load_pubmed_documents

_vector_store = None


def get_vector_store(force_rebuild=False):
    """Load FAISS index from disk, or build and save it if missing."""
    global _vector_store
    if _vector_store is not None and not force_rebuild:
        return _vector_store

    index_dir = Path(FAISS_INDEX_PATH)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_exists = (index_dir / "index.faiss").exists() or (index_dir / "index.pkl").exists()
    embeddings = get_embeddings()

    if not force_rebuild and index_exists:
        _vector_store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return _vector_store

    documents = load_pubmed_documents()
    _vector_store = FAISS.from_documents(documents, embeddings)
    _vector_store.save_local(FAISS_INDEX_PATH)
    return _vector_store


def get_retriever(force_rebuild=False):
    """Return a retriever over the FAISS index (similarity search, top k)."""
    store = get_vector_store(force_rebuild=force_rebuild)
    return store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": RETRIEVER_K},
    )
