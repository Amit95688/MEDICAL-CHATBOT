"""Simple retriever using FAISS."""
from pathlib import Path
from langchain_community.vectorstores import FAISS
from app.config import FAISS_INDEX_PATH, RETRIEVER_K
from app.rag.embeddings import get_embeddings
from app.rag.loader import load_pubmed_documents


def get_vector_store():
    """Load FAISS index or create it."""
    index_dir = Path(FAISS_INDEX_PATH)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if index exists
    if (index_dir / "index.faiss").exists():
        print(f"Loading index from {FAISS_INDEX_PATH}")
        embeddings = get_embeddings()
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    
    # Build new index
    print(f"Building new FAISS index...")
    documents = load_pubmed_documents()
    embeddings = get_embeddings()
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(FAISS_INDEX_PATH)
    print(f"Index saved to {FAISS_INDEX_PATH}")
    return store


def get_retriever():
    """Return retriever (similarity search, top k)."""
    store = get_vector_store()
    print(f"Retrieving top {RETRIEVER_K} documents")
    return store.as_retriever(search_kwargs={"k": RETRIEVER_K})
