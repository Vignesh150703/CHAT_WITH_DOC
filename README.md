# Vector DB RAG Chat (Pinecone + Groq + Streamlit)

Streamlit app to upload documents, index them into **Pinecone**, and chat with them using **Groq** (RAG: Retrieval-Augmented Generation).  
It supports document-wise filtering (per uploaded file), semantic-ish chunking, cross-encoder reranking, and token-budgeted prompting to avoid oversized LLM requests.

---

## Features

- **Multi-file upload**: PDF, DOCX, TXT, images (`png/jpg/jpeg`) and other formats via Unstructured.
- **Chunk + embed + upsert to Pinecone** (deduped with deterministic vector IDs).
- **Document filter dropdown**: choose “All documents” or a specific uploaded document.
- **High-precision retrieval**:
  - Dense vector search in Pinecone
  - Lightweight keyword score
  - **Cross-encoder reranking** for on-point context
- **Citation-friendly answers**: context is formatted with chunk numbers like `[1]`, `[2]`.
- **Groq request-size protection**: token-budgeted context to reduce 413/TPM errors.

---

## Tech stack

- **UI**: Streamlit
- **RAG framework**: LangChain (core/community)
- **Vector DB**: Pinecone (`global` namespace)
- **LLM**: Groq via `langchain-groq` (default model in code: `llama-3.1-8b-instant`)
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2` via `HuggingFaceEmbeddings`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` `CrossEncoder`
- **Chunking**:
  - Primary: spaCy sentence segmentation (`en_core_web_sm`)
  - Fallback: `RecursiveCharacterTextSplitter.from_tiktoken_encoder`
- **Parsing / loaders**:
  - PDF: `PyPDFLoader` (pypdf)
  - DOCX: `Docx2txtLoader`
  - TXT: `TextLoader`
  - Other / images: `UnstructuredLoader` (`unstructured`, `langchain-unstructured`)
- **Token budgeting**: `tiktoken`

---

## Repository structure

- `pinecone/app.py`: Streamlit application (indexing + retrieval + chat).
- `pinecone/requirements.txt`: Python dependencies for the app.
- `pinecone/doc_catalog.json`: Persistent list of document names for the dropdown (auto-generated).
- `.env`: Environment variables (you create locally; do not commit secrets).

---

## Setup

### 1) Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r .\pinecone\requirements.txt
```

### 3) Install spaCy model (required for sentence-based chunking)

```powershell
python -m spacy download en_core_web_sm
```

If you skip this, the app will automatically fall back to a token-based text splitter.

### 4) Configure environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY="YOUR_PINECONE_KEY"
GROQ_API_KEY="YOUR_GROQ_KEY"
```

---

## Run the app

From the project root:

```powershell
streamlit run .\pinecone\app.py
```

Open the displayed local URL in your browser.

---

## How it works (architecture)

### Indexing flow (Upload → Pinecone)

1. **Upload files** in the sidebar.
2. Click **Process documents**.
3. The app:
   - Loads files into LangChain `Document`s using format-specific loaders.
   - Adds metadata (notably `source_file` and `page`).
   - Chunks content using **spaCy sentence segmentation** into ~4k-character chunks with overlap.
   - Embeds chunks using MPNet (`all-mpnet-base-v2`).
   - Upserts vectors into Pinecone under namespace **`global`** with deterministic IDs (dedupe).
4. Document names are stored in `pinecone/doc_catalog.json` to populate the dropdown next run.

### Query flow (Question → Answer)

1. User enters a question.
2. `retrieve_with_scores()`:
   - Runs Pinecone similarity search (optionally filtered by `source_file`).
   - De-duplicates near-identical hits.
   - Applies a preliminary score (dense + keyword overlap).
   - **Reranks** candidates with a cross-encoder to select the most relevant chunks.
3. `format_cited_context()`:
   - Formats the top chunks into a context block with citations `[1]`, `[2]`, …
   - Enforces a token budget to reduce Groq 413/TPM errors.
4. Groq LLM answers using a strict prompt:
   - “Use only provided context”
   - “Quote-first for exact values”
   - “Cite evidence with chunk numbers”

---

## Configuration knobs (in code)

In `pinecone/app.py`:

- **Retrieval defaults**:
  - `DENSE_K_DEFAULT`: how many vectors to fetch from Pinecone per query
  - `FINAL_K_DEFAULT`: how many reranked chunks to pass to the LLM
  - `RERANK_CANDIDATES_MAX`: how many candidates are reranked
- **Context budget**:
  - `format_cited_context(..., max_context_tokens=3400)`
- **Chunking**:
  - spaCy-based chunking uses `max_chars = 4000` and `overlap_sents = 2`

---

## Troubleshooting

### Groq error: 413 / “Request too large” / TPM limit exceeded

Cause: prompt + retrieved context exceeds Groq limits.

Fixes:
- Reduce `max_context_tokens` in `format_cited_context()` (e.g., 3400 → 2800).
- Reduce `FINAL_K_DEFAULT` (fewer chunks sent to the LLM).
- Keep reranking enabled so fewer, higher-quality chunks are needed.

### spaCy error: model `en_core_web_sm` not found

Run:

```powershell
python -m spacy download en_core_web_sm
```

The app also falls back to a token-based splitter if spaCy isn’t available.

### Pinecone index dimension mismatch

Your embeddings are **768-dimensional**. If your Pinecone index was created with a different dimension, the code may delete and recreate it.

Important:
- Recreating the index deletes existing vectors.
- Ensure you intend that before running against production data.

### Document filter dropdown only shows “All documents”

The dropdown list comes from `pinecone/doc_catalog.json` (written when you process uploads).  
If you haven’t processed documents in this workspace yet, the list will be empty.

---

## Security notes

- Do **not** commit `.env` files or API keys.
- Consider removing any hard-coded fallback keys from `pinecone/app.py` if present in your environment.

---

## License

Add your preferred license (MIT/Apache-2.0/etc.).

