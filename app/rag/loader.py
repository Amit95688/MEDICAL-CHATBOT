"""Load PubMed dataset and split into chunks (transferred from notebook)."""
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import (
    DATASET_NAME,
    DATASET_SPLIT,
    DATASET_SAMPLE_SIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_pubmed_documents():
    """Load PubMed summarization dataset and return list of LangChain Documents."""
    dataset = load_dataset(
        DATASET_NAME,
        split=f"{DATASET_SPLIT}[:{DATASET_SAMPLE_SIZE}]",
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_chunks = []
    for item in dataset:
        combined_text = item["abstract"] + "\n\n" + item["article"]
        chunks = splitter.create_documents([combined_text])
        all_chunks.extend(chunks)
    return all_chunks
