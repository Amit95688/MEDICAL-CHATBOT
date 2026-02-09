"""Simple embeddings module."""
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE


def get_embeddings():
    """Get embeddings model."""
    print(f"Loading embeddings: {EMBEDDING_MODEL_NAME} on {EMBEDDING_DEVICE}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )
