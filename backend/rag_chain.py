"""
SWS AI RAG Chatbot — RAG Chain
--------------------------------
Builds a LangChain RetrievalQA chain using:
  • ChromaDB  — vector store
  • OpenAI    — embeddings (text-embedding-3-small)
  • Anthropic — LLM (claude-3-5-sonnet)
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR  = str(Path(__file__).parent / "chroma_db")
COLLECTION  = "sws_ai_docs"
RETRIEVAL_K = 5          # top-k chunks retrieved per query

# ── System Prompt ─────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are the SWS AI internal HR & Policy Assistant.
Your ONLY knowledge source is the official company policy documents provided below.

RULES:
1. Answer STRICTLY from the context below — do not use outside knowledge.
2. If the answer is not found in the context, respond EXACTLY:
   "I don't have that information in the company documents."
3. Be concise, clear, and professional.
4. When relevant, mention the specific policy section or document name.
5. Do not fabricate numbers, dates, or policy details.

─────────────────────────────────────────────────────────────────
CONTEXT FROM COMPANY DOCUMENTS:
{context}
─────────────────────────────────────────────────────────────────

EMPLOYEE QUESTION: {question}

ANSWER:"""


def get_rag_chain():
    """Build and return the RAG retrieval chain."""

    # 1. Embeddings (same model used during ingestion)
    embeddings = OpenAIEmbeddings(
        model   = "text-embedding-3-small",
        api_key = os.getenv("OPENAI_API_KEY"),
    )

    # 2. ChromaDB vector store
    vectorstore = Chroma(
        persist_directory = CHROMA_DIR,
        embedding_function= embeddings,
        collection_name   = COLLECTION,
    )

    # 3. Retriever — fetch top-k semantically relevant chunks
    retriever = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": RETRIEVAL_K},
    )

    # 4. Anthropic Claude LLM
    llm = ChatAnthropic(
        model       = "claude-3-5-sonnet-20241022",
        temperature = 0,           # deterministic answers
        max_tokens  = 1024,
        api_key     = os.getenv("ANTHROPIC_API_KEY"),
    )

    # 5. Prompt
    prompt = PromptTemplate(
        template        = PROMPT_TEMPLATE,
        input_variables = ["context", "question"],
    )

    # 6. RetrievalQA chain (stuff = concat all chunks into one prompt)
    chain = RetrievalQA.from_chain_type(
        llm                  = llm,
        chain_type           = "stuff",
        retriever            = retriever,
        return_source_documents = True,
        chain_type_kwargs    = {"prompt": prompt},
    )

    return chain
