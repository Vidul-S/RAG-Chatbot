"""
SWS AI RAG Chatbot — Document Ingestion Pipeline
-------------------------------------------------
Loads PDFs from /documents, chunks them, generates embeddings,
and stores everything in ChromaDB.

Run: python ingest.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ── Config ────────────────────────────────────────────────────────────────────
DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"
CHROMA_DIR    = str(Path(__file__).parent / "chroma_db")
COLLECTION    = "sws_ai_docs"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


def ingest_documents():
    print("\n" + "═" * 55)
    print("  SWS AI — Document Ingestion Pipeline")
    print("═" * 55)

    # ── 1. Load PDFs ──────────────────────────────────────────
    print(f"\n📂 Scanning: {DOCUMENTS_DIR}")
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("❌  No PDF files found. Add PDFs to the /documents folder.")
        sys.exit(1)

    documents = []
    for pdf_path in pdf_files:
        print(f"   📄 Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs   = loader.load()

        # Attach metadata: source filename + page number
        for i, doc in enumerate(docs):
            doc.metadata["source"]     = pdf_path.name
            doc.metadata["doc_title"]  = pdf_path.stem.replace("_", " ").replace("-", " ").title()
            doc.metadata["chunk_index"] = i

        documents.extend(docs)

    print(f"\n✅  Loaded {len(documents)} pages from {len(pdf_files)} documents")

    # ── 2. Chunk ──────────────────────────────────────────────
    print(f"\n✂️   Chunking  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅  Created {len(chunks)} chunks")

    # ── 3. Embed + Store ──────────────────────────────────────
    print("\n🧠  Generating embeddings (OpenAI text-embedding-3-small)...")
    embeddings = OpenAIEmbeddings(
        model     = "text-embedding-3-small",
        api_key   = os.getenv("OPENAI_API_KEY"),
    )

    # Wipe existing collection so re-ingestion is clean
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
        print("   ♻️  Cleared existing collection")
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents        = chunks,
        embedding        = embeddings,
        persist_directory= CHROMA_DIR,
        collection_name  = COLLECTION,
    )

    print(f"✅  Stored {len(chunks)} chunks in ChromaDB → {CHROMA_DIR}")

    # ── 4. Quick sanity test ──────────────────────────────────
    print("\n🔍  Sanity test — querying: 'What is the leave policy?'")
    results = vectorstore.similarity_search("What is the leave policy?", k=3)
    for r in results:
        print(f"   ↳ [{r.metadata.get('source', 'unknown')}] {r.page_content[:80]}…")

    print("\n🎉  Ingestion complete! You can now run the API.\n")


if __name__ == "__main__":
    ingest_documents()
