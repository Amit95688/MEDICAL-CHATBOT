"""
Build the FAISS index once. Run this before starting the app for faster startup.

  python -m scripts.build_index

Or from project root:
  PYTHONPATH=. python scripts/build_index.py

After this, the app will load the index from disk instead of building on first request.
"""
import sys
from pathlib import Path

# Add project root so "app" is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from app.rag.retriever import get_vector_store


def main():
    print("Building FAISS index (dataset + embeddings)...")
    get_vector_store(force_rebuild=True)
    print("Done. Index saved. Start the app and queries will be fast.")


if __name__ == "__main__":
    main()
