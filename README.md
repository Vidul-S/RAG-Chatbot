# SWS AI RAG Chatbot

> An internal HR & Policy Q&A chatbot powered by Retrieval-Augmented Generation (RAG).  
> Employees can ask natural language questions and receive accurate, grounded answers sourced directly from company policy documents.

---

## Demo

| AI Assistant | Document Upload |
|---|---|
| Ask policy questions in natural language | Upload & manage company PDFs |
| Sources shown per answer | Drag-and-drop with progress bars |
| Typing indicator while processing | Bulk notification flow |

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| **LLM** | Anthropic Claude 3.5 Sonnet | Best-in-class reasoning, strict instruction following |
| **Embeddings** | OpenAI `text-embedding-3-small` | High quality, cost-efficient, 1536 dimensions |
| **Vector DB** | ChromaDB (local, persistent) | Zero setup, perfect for prototypes, file-based persistence |
| **Framework** | LangChain | Best RAG ecosystem, RetrievalQA chain, document loaders |
| **Backend** | FastAPI | Async Python, clean REST API, auto docs at `/docs` |
| **Frontend** | Plain HTML/CSS/JS | No build step, matches SWS AI design (Livvic font, blue/white) |

---

## Project Structure

```
sws-ai-rag-chatbot/
├── backend/
│   ├── main.py            # FastAPI app — POST /api/chat endpoint
│   ├── ingest.py          # PDF ingestion pipeline
│   ├── rag_chain.py       # LangChain RAG chain (retriever + LLM)
│   ├── requirements.txt
│   ├── .env.example       # Copy to .env and fill in keys
│   └── chroma_db/         # Auto-created after ingestion (gitignored)
├── frontend/
│   └── index.html         # Complete chat UI
├── documents/             # ← Put your 10 PDF files here
│   └── README.txt
├── .gitignore
└── README.md
```

---

## Setup & Running

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sws-ai-rag-chatbot.git
cd sws-ai-rag-chatbot
```

### 2. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Add company PDF documents

Place all 10 SWS AI company PDF files inside the `/documents` folder:

```
documents/
├── HR_Policy.pdf
├── Leave_Policy.pdf
├── Resignation_Policy.pdf
├── IT_Security_Policy.pdf
└── ... (all 10 PDFs)
```

### 5. Run the ingestion pipeline

```bash
cd backend
python ingest.py
```

This will:
- Load and parse all PDFs
- Chunk them into 500-token segments with 50-token overlap
- Generate OpenAI embeddings for each chunk
- Store everything in ChromaDB at `backend/chroma_db/`
- Run a quick sanity test query

Expected output:
```
══════════════════════════════════════════════
  SWS AI — Document Ingestion Pipeline
══════════════════════════════════════════════

📂 Scanning: /documents
   📄 Loading: HR_Policy.pdf
   📄 Loading: Leave_Policy.pdf
   ...

✅  Loaded 87 pages from 10 documents
✂️   Chunking  (size=500, overlap=50)...
✅  Created 312 chunks
🧠  Generating embeddings (OpenAI text-embedding-3-small)...
✅  Stored 312 chunks in ChromaDB

🎉  Ingestion complete!
```

### 6. Start the FastAPI backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API is now running at `http://localhost:8000`  
Auto-generated docs: `http://localhost:8000/docs`

### 7. Open the Chat UI

Open `frontend/index.html` in your browser — or visit `http://localhost:8000` (FastAPI serves the frontend too).

---

## API Reference

### `POST /api/chat`

Accepts a question, retrieves relevant document chunks, and returns a grounded answer.

**Request:**
```json
{
  "question": "How many sick leaves do I get?"
}
```

**Response:**
```json
{
  "answer": "According to the Leave Policy, employees are entitled to 10 sick leave days per year...",
  "sources": ["Leave Policy", "HR Policy"],
  "duration": 1.84
}
```

### `GET /api/health`

```json
{ "status": "ok", "model": "claude-3-5-sonnet-20241022" }
```

### `GET /api/docs-list`

Lists all ingested PDF files.

---

## Architecture Decisions

### 1. Chunking Strategy

| Parameter | Value | Reasoning |
|---|---|---|
| `chunk_size` | 500 tokens | Large enough to contain a complete policy clause, small enough to stay focused |
| `chunk_overlap` | 50 tokens | Prevents context loss at chunk boundaries (e.g., a sentence split across chunks) |
| Splitter | `RecursiveCharacterTextSplitter` | Tries paragraph → newline → sentence → word splits in order, preserving natural boundaries |
| Separators | `["\n\n", "\n", ". ", " ", ""]` | Respects document structure (sections, paragraphs) before falling back to words |

**Why 500 tokens?** Policy documents use dense, structured language. At 500 tokens we capture full policy clauses without diluting the retrieval signal with unrelated content.

---

### 2. Embedding Model — `text-embedding-3-small`

- **Dimensions:** 1,536
- **Why chosen:** Best balance of quality vs. cost for semantic search over structured text
- **Consistency:** Same model used for both ingestion and query embedding — critical for accurate similarity search
- **Alternatives considered:** `all-MiniLM-L6-v2` (free, local) — chosen OpenAI instead for higher accuracy on formal document language

---

### 3. Vector Database — ChromaDB

- **Type:** Local, file-based, persistent
- **Why chosen:** Zero infrastructure setup, Python-native, perfect for this use case
- **Collection:** `sws_ai_docs` — all 10 documents in one collection
- **Metadata stored per chunk:** `source` (filename), `doc_title` (human-readable name), `page_number`, `chunk_index`
- **Alternatives considered:** FAISS (in-memory, no persistence), Pinecone (cloud, requires account)

---

### 4. Retrieval K Value

```python
retrieval_k = 5
```

- Retrieves the **top 5 most semantically similar chunks** per query
- **Why 5?** Balances recall (finding the right answer) vs. context window size (too many chunks = diluted prompt)
- **Search type:** Cosine similarity (ChromaDB default)
- For broad questions (e.g., "tell me all about leave"), 5 chunks from multiple documents gives comprehensive coverage

---

### 5. Prompt Design

The system prompt enforces strict grounding:

```
You are the SWS AI internal HR & Policy Assistant.
Your ONLY knowledge source is the official company policy documents provided below.

RULES:
1. Answer STRICTLY from the context below — do not use outside knowledge.
2. If the answer is not found in the context, respond EXACTLY:
   "I don't have that information in the company documents."
3. Be concise, clear, and professional.
4. When relevant, mention the specific policy section or document name.
5. Do not fabricate numbers, dates, or policy details.
```

**Key design choices:**
- `temperature=0` — fully deterministic answers, no hallucination risk
- Explicit fallback phrase when answer is not in documents
- "STRICTLY from the context" — prevents Claude from using general world knowledge
- Chain type: `stuff` — all retrieved chunks concatenated into one prompt (works well for k=5 at 500 tokens each)

---

## Sample Queries to Demo

| Question | Expected Source |
|---|---|
| How many sick leave days do I get? | Leave Policy |
| What is the notice period for resignation? | Resignation Policy |
| What are the WFH guidelines? | WFH Guidelines |
| What is the IT password policy? | IT Security Policy |
| How does the performance review work? | Performance Review Policy |
| What health insurance benefits do we have? | Benefits Policy |

---

## GitHub Commit Strategy

Commits are made at least every 15 minutes following this pattern:

```
feat: initial project structure
feat: add PDF ingestion pipeline (ingest.py)
feat: add ChromaDB vector store setup
feat: add LangChain RAG chain with Claude
feat: add FastAPI /api/chat endpoint
feat: add chat UI with source display
feat: add document upload tab
fix: handle empty documents folder gracefully
docs: add README with architecture decisions
```

---

## Limitations & Future Improvements

- **No authentication** — add OAuth for production use
- **Single collection** — could partition by department
- **No re-ranking** — add cross-encoder re-ranking for higher accuracy
- **No streaming** — Claude streaming responses for faster perceived latency
- **PDF images** — text-only extraction; scanned PDFs need OCR (pytesseract)
