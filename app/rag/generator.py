"""
Answer generation: fast path (retrieve only) and full RAG (retrieve + LLM).

- Fast path: no LLM, just format retrieved chunks. Fast and avoids device errors.
- Full path: optional LLM with explicit device to fix "Tensor on device meta" errors.
"""
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.config import (
    USE_FAST_PATH,
    LLM_MODEL_NAME,
    LLM_DEVICE,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)
from app.rag.retriever import get_retriever


# -----------------------------------------------------------------------------
# Helpers: format docs and short summary
# -----------------------------------------------------------------------------

def format_docs(docs):
    """Turn retrieved chunks into one string."""
    return "\n\n".join(doc.page_content for doc in docs)


def extract_summary(text, num_sentences=3):
    """First n sentences as a short summary."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return ". ".join(sentences[:num_sentences]) + "." if sentences else text[:200]


def format_for_summarization(retrieved_docs):
    """Top 2 docs, then short summary."""
    combined = format_docs(retrieved_docs[:2])
    return extract_summary(combined)


# -----------------------------------------------------------------------------
# Fast path: retrieve + format only (no LLM) — fast and stable
# -----------------------------------------------------------------------------

def query_fast(question: str) -> str:
    """
    Return answer using only retrieval: get top chunks and format them.
    No LLM, so it's quick and avoids GPU/device errors.
    """
    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = format_docs(docs)
    if not context.strip():
        return "No relevant context found. Try rephrasing or a different question."
    return (
        "Based on the retrieved medical context:\n\n"
        + context
        + "\n\n---\n(Answer from retrieved context only. For a generated summary, use /api/summarize.)"
    )


def query_summary(question: str) -> str:
    """Retrieve and return a short summary (first few sentences of top chunks)."""
    retriever = get_retriever()
    docs = retriever.invoke(question)
    return format_for_summarization(docs)


# -----------------------------------------------------------------------------
# Full RAG: retrieve + LLM (optional, slower)
# -----------------------------------------------------------------------------

MEDICAL_TEMPLATE = """
You are a professional medical assistant AI.

Guidelines:
- Use ONLY the provided text as context.
- If the information is not in the text, respond: "Not available in the provided context. Please consult a medical professional."
- Give concise, accurate, and neutral answers.
- Avoid guessing or providing unverified information.

Medical Context:
{context}

Patient Question:
{question}

Answer:
"""

medical_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=MEDICAL_TEMPLATE,
)

_llm = None
_full_chain = None


def _get_llm():
    """
    Load LLM with explicit device to avoid "Tensor on device meta is not on the expected device cuda:0".
    Uses device from config (default: cpu) and no device_map so all tensors are on one device.
    """
    global _llm
    if _llm is not None:
        return _llm

    device = torch.device(LLM_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load without device_map so nothing stays on "meta"; then move to device
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
    )
    _llm = HuggingFacePipeline(pipeline=pipe)
    return _llm


def _get_full_chain():
    """RAG chain: question -> retrieve -> prompt -> LLM -> answer."""
    global _full_chain
    if _full_chain is not None:
        return _full_chain

    retriever = get_retriever()
    llm = _get_llm()

    _full_chain = (
        RunnableLambda(lambda x: x if isinstance(x, str) else x.get("question", x))
        | {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | medical_prompt
        | llm
        | StrOutputParser()
    )
    return _full_chain


def query_full_rag(question: str) -> str:
    """Full RAG with LLM. Slower; use when you need a generated answer."""
    return _get_full_chain().invoke(question)


# -----------------------------------------------------------------------------
# Main entry: use fast path by default for speed
# -----------------------------------------------------------------------------

def query_rag(question: str, use_fast: bool = None) -> str:
    """
    Answer a question. By default uses fast path (retrieve only).
    Set use_fast=False or USE_FAST_PATH=0 to use full RAG with LLM.
    """
    if use_fast is None:
        use_fast = USE_FAST_PATH
    if use_fast:
        return query_fast(question)
    return query_full_rag(question)
