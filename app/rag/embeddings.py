"""Embeddings model (transferred from notebook: HuggingFace all-MiniLM-L6-v2)."""
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE


def get_embeddings():
    """Return HuggingFace embeddings instance used in the notebook."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )
