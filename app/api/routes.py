"""
API routes for the Medical RAG app.

- /api/health     — health check
- /api/query      — fast answer (retrieve only, default)
- /api/query_full — full RAG with LLM (slower)
- /api/summarize  — short summary of retrieved chunks
"""
from flask import Blueprint, request, jsonify

from app.rag.generator import query_fast, query_full_rag, query_summary

api = Blueprint("api", __name__, url_prefix="/api")


def _get_question():
    """Read question from JSON body or form."""
    data = request.get_json(silent=True) or {}
    q = data.get("question")
    if not q and request.form:
        q = request.form.get("question")
    return (q or "").strip()


@api.route("/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({"status": "ok", "service": "medical-chatbot"})


@api.route("/query", methods=["POST"])
def query():
    """
    Fast answer: retrieve + format only (no LLM).
    Quick and avoids device/GPU errors.
    """
    question = _get_question()
    if not question:
        return jsonify({"error": "Missing or empty 'question'"}), 400
    try:
        answer = query_fast(question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route("/query_full", methods=["POST"])
def query_full():
    """
    Full RAG: retrieve + LLM generation.
    Slower; use when you need a generated answer.
    """
    question = _get_question()
    if not question:
        return jsonify({"error": "Missing or empty 'question'"}), 400
    try:
        answer = query_full_rag(question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route("/summarize", methods=["POST"])
def summarize():
    """Short summary of retrieved chunks (no LLM)."""
    question = _get_question()
    if not question:
        return jsonify({"error": "Missing or empty 'question'"}), 400
    try:
        summary = query_summary(question)
        return jsonify({"question": question, "summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
