"""Simple API routes for the Medical RAG app."""
from flask import Blueprint, request, jsonify
from app.rag.generator import query

api = Blueprint("api", __name__, url_prefix="/api")


def get_question():
    """Get question from JSON body."""
    data = request.get_json(silent=True) or {}
    return (data.get("question") or "").strip()


@api.route("/health", methods=["GET"])
def health():
    """Health check."""
    print("Health check")
    return jsonify({"status": "ok"})


@api.route("/query", methods=["POST"])
def answer_question():
    """Answer a medical question using retrieved documents."""
    question = get_question()
    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    try:
        print(f"Answering: {question}")
        answer = query(question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
