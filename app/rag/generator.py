"""Simple answer generation using retrieval."""
from app.rag.retriever import get_retriever


def format_docs(docs):
    """Format retrieved documents into text."""
    return "\n\n".join(doc.page_content for doc in docs)


def query(question: str) -> str:
    """Get answer for a question using retrieved documents."""
    print(f"\nQuery: {question}")
    
    retriever = get_retriever()
    docs = retriever.invoke(question)
    
    if not docs:
        print("No documents found")
        return "No relevant documents found."
    
    print(f"Found {len(docs)} documents")
    answer = format_docs(docs)
    return answer
