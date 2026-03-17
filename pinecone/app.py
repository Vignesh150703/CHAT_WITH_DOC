"""
Streamlit app to chat with uploaded documents using Pinecone + Groq.

Features:
- Upload one or more documents (PDF, DOC/DOCX, images, text).
- Chunk and embed the documents with HuggingFaceEmbeddings.
- Store / reuse vectors in Pinecone.
- Chat UI (Streamlit chat) to ask questions; Groq LLM answers using retrieved context.
"""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import spacy
import streamlit as st
import tiktoken
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import CrossEncoder


load_dotenv()

DEFAULT_INDEX_NAME = "pinecone"
DEFAULT_NAMESPACE = "global"
DOC_CATALOG_PATH = "doc_catalog.json"

# Retrieval / reranking defaults (centralized to avoid magic numbers)
DENSE_K_DEFAULT = 60
FINAL_K_DEFAULT = 10
RERANK_CANDIDATES_MAX = 50
RERANK_TEXT_CHARS = 1800
RERANK_TIEBREAK_WEIGHT = 0.01


# ---------- Helpers ----------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stable_chunk_id(namespace: str, source_file: str, page: int, chunk_index: int, text: str) -> str:
    """Deterministic vector ID so re-indexing doesn't create duplicates."""
    base = f"{namespace}|{source_file}|{page}|{chunk_index}|{text}".encode("utf-8", errors="ignore")
    return sha256_bytes(base)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def keyword_score(query: str, text: str) -> float:
    """Lightweight keyword overlap score (works well for exact headings/dates/tables)."""
    q = set(tokenize(query))
    if not q:
        return 0.0
    t = set(tokenize(text))
    return len(q.intersection(t)) / max(1, len(q))


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    # Stable tokenizer for rough token budgeting (prevents oversized Groq requests)
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    enc = get_tokenizer()
    return len(enc.encode(text or ""))


@dataclass
class RetrievedChunk:
    doc: object
    dense_score: float
    kw_score: float
    rerank_score: float
    final_score: float


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    # Explicit model for stability across environments
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGroq:
    return init_groq_llm()


@st.cache_resource(show_spinner=False)
def get_reranker() -> CrossEncoder:
    # Fast, strong general-purpose cross-encoder for reranking
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner=False)
def get_spacy_nlp():
    # Requires: python -m spacy download en_core_web_sm
    return spacy.load("en_core_web_sm")


def init_pinecone(index_name: str = DEFAULT_INDEX_NAME, dim: int = 768) -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set. Add it to your .env file.")

    pc = Pinecone(api_key=api_key)

    existing_indexes = pc.list_indexes().names()

    # If index exists but with wrong dimension, delete and recreate it
    if index_name in existing_indexes:
        desc = pc.describe_index(index_name)
        existing_dim = desc.dimension
        if existing_dim != dim:
            pc.delete_index(index_name)
            existing_indexes = [name for name in existing_indexes if name != index_name]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc


def add_documents_deduped(
    vectorstore: LangchainPinecone,
    docs,
) -> int:
    """Add documents with deterministic IDs so re-uploads don't duplicate."""
    ids: List[str] = []
    metadatas = []
    texts = []
    for i, d in enumerate(docs):
        md = dict(getattr(d, "metadata", {}) or {})
        source_file = str(md.get("source_file") or md.get("source") or "uploaded")
        page = int(md.get("page", 1))
        chunk_index = int(md.get("chunk_index", i))
        text = str(getattr(d, "page_content", "") or "")
        _id = stable_chunk_id(DEFAULT_NAMESPACE, source_file, page, chunk_index, text)
        md["namespace"] = DEFAULT_NAMESPACE
        md["chunk_index"] = chunk_index
        ids.append(_id)
        metadatas.append(md)
        texts.append(text)

    # LangChain Pinecone wrapper supports add_texts with ids/namespace
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids, namespace=DEFAULT_NAMESPACE)
    return len(ids)


def load_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    """Save uploaded files to temp paths and load as LangChain Documents.

    Uses local loaders for common types and falls back to UnstructuredLoader
    for others. Any loader errors are shown in the UI instead of crashing
    the Streamlit server.

    Returns:
        List of Document objects loaded from all files
    """
    all_docs = []
    failed_files = []

    for uf in uploaded_files:
        suffix = "." + uf.name.split(".")[-1] if "." in uf.name else ""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name

            ext = suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif ext == ".docx":
                loader = Docx2txtLoader(tmp_path)
            else:
                # Fallback for images and other formats
                loader = UnstructuredLoader(tmp_path)

            docs = loader.load()
            if docs:
                # Attach helpful metadata (file name, page) for better retrieval signal
                for i, d in enumerate(docs):
                    d.metadata.setdefault("source_file", uf.name)
                    if "page" not in d.metadata:
                        d.metadata["page"] = i + 1
                all_docs.extend(docs)
            else:
                failed_files.append(f"{uf.name} (no content extracted)")
        except Exception as e:
            failed_files.append(f"{uf.name}: {str(e)}")
            continue
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:  # pragma: no cover - best-effort cleanup
                    pass

    # Show warnings for failed files but don't fail completely
    if failed_files:
        st.warning(f"⚠️ Some files could not be loaded: {', '.join(failed_files)}")

    return all_docs


def chunk_docs(docs, chunk_size: int = 900, chunk_overlap: int = 200):
    """
    Semantic chunking:
    - Use spaCy sentence segmentation to get sentence-level units.
    - Group sentences into chunks up to a target character budget (~token budget proxy).
    - Add light sentence overlap between chunks to preserve context.
    Falls back to tiktoken-based splitter if spaCy model is unavailable.
    """
    try:
        nlp = get_spacy_nlp()
    except Exception:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents(docs)

    max_chars = 4000  # ~900–1000 tokens depending on text
    overlap_sents = 2
    out: List[Document] = []

    for d in docs:
        text = getattr(d, "page_content", "") or ""
        if not text.strip():
            continue
        doc_nlp = nlp(text)
        sents = [s.text.strip() for s in doc_nlp.sents if s.text.strip()]
        if not sents:
            continue

        current_sents: List[str] = []
        current_len = 0

        for sent in sents:
            s_len = len(sent)
            if current_sents and current_len + 1 + s_len > max_chars:
                # Flush current chunk
                chunk_text = " ".join(current_sents).strip()
                if chunk_text:
                    out.append(
                        Document(
                            page_content=chunk_text,
                            metadata=dict(getattr(d, "metadata", {}) or {}),
                        )
                    )
                # Start new chunk with overlap
                overlap = current_sents[-overlap_sents:] if len(current_sents) > overlap_sents else current_sents
                current_sents = overlap + [sent]
                current_len = len(" ".join(current_sents))
            else:
                current_sents.append(sent)
                current_len += (1 if current_sents else 0) + s_len

        # Flush tail
        if current_sents:
            chunk_text = " ".join(current_sents).strip()
            if chunk_text:
                out.append(
                    Document(
                        page_content=chunk_text,
                        metadata=dict(getattr(d, "metadata", {}) or {}),
                    )
                )

    return out


def init_groq_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )


def retrieve_with_scores(
    query: str,
    vectorstore: LangchainPinecone,
    *,
    doc_name_filter: Optional[str] = None,
    dense_k: int = DENSE_K_DEFAULT,
    final_k: int = FINAL_K_DEFAULT,
) -> List[RetrievedChunk]:
    """Dense retrieval + lightweight keyword score + cross-encoder rerank."""
    # Keep retrieval simple and let the cross-encoder reranker do the heavy lifting.
    dense_results: List[Tuple[object, float]] = []
    # Build Pinecone metadata filter for document-wise retrieval, if requested
    search_kwargs = {}
    if doc_name_filter:
        search_kwargs["filter"] = {"source_file": {"$eq": doc_name_filter}}

    dense_results.extend(
        vectorstore.similarity_search_with_score(
            query, k=dense_k, namespace=DEFAULT_NAMESPACE, **search_kwargs
        )
    )

    # De-duplicate docs by content hash (keep best dense score)
    best_by_hash: Dict[str, Tuple[object, float]] = {}
    for doc, score in dense_results:
        text = (getattr(doc, "page_content", "") or "").strip()
        h = sha256_bytes(text.encode("utf-8", errors="ignore"))
        if h not in best_by_hash or float(score) > float(best_by_hash[h][1]):
            best_by_hash[h] = (doc, float(score))

    dense = list(best_by_hash.values())

    # Score candidates (dense + keyword); then rerank the best subset with a cross-encoder
    prelim: List[RetrievedChunk] = []
    for doc, score in dense:
        # Pinecone/LangChain score semantics can vary; treat as dense relevance score
        text = getattr(doc, "page_content", "") or ""
        kw = keyword_score(query, text)
        prelim_score = (0.85 * float(score)) + (0.15 * kw)
        prelim.append(
            RetrievedChunk(
                doc=doc,
                dense_score=float(score),
                kw_score=kw,
                rerank_score=0.0,
                final_score=prelim_score,
            )
        )

    prelim.sort(key=lambda x: x.final_score, reverse=True)

    # Cross-encoder rerank on a capped candidate set (big accuracy lift, controlled latency)
    candidates = prelim[: min(RERANK_CANDIDATES_MAX, len(prelim))]
    try:
        reranker = get_reranker()
        pairs = []
        for ch in candidates:
            txt = (getattr(ch.doc, "page_content", "") or "").strip()
            pairs.append((query, txt[:RERANK_TEXT_CHARS]))
        rr_scores = reranker.predict(pairs)
        for ch, rr in zip(candidates, rr_scores):
            ch.rerank_score = float(rr)
            # Primary ranking uses reranker; keep a tiny tie-break from prelim score
            ch.final_score = (1.0 * ch.rerank_score) + (RERANK_TIEBREAK_WEIGHT * ch.final_score)
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:final_k]
    except Exception:
        # If reranker is unavailable for any reason, fall back to prelim ranking
        return prelim[:final_k]


def format_cited_context(chunks: List[RetrievedChunk], *, max_context_tokens: int = 3400) -> str:
    """
    Build a cited context string while staying under a token budget.
    This prevents Groq 413 (request too large / TPM) errors.
    """
    parts: List[str] = []
    used = 0
    for i, ch in enumerate(chunks, 1):
        doc = ch.doc
        md = getattr(doc, "metadata", {}) or {}
        src = md.get("source_file") or md.get("source") or "unknown"
        page = md.get("page", "?")
        text = (getattr(doc, "page_content", "") or "").strip()
        # Cap per-chunk text so one huge chunk can't blow the budget
        text = text[:3500]
        part = f"[{i}] Source: {src} | Page: {page}\n{text}"
        part_tokens = count_tokens(part)
        if parts and (used + part_tokens) > max_context_tokens:
            break
        parts.append(part)
        used += part_tokens
    return "\n\n".join(parts)


def build_context_stats(query: str, chunks: List[RetrievedChunk]) -> Dict[str, object]:
    source_files = sorted(
        {
            (getattr(ch.doc, "metadata", {}) or {}).get("source_file")
            or (getattr(ch.doc, "metadata", {}) or {}).get("source")
            or "unknown"
            for ch in chunks
        }
    )
    combined_text = " ".join((getattr(ch.doc, "page_content", "") or "") for ch in chunks)
    overlap = keyword_score(query, combined_text)
    return {
        "num_chunks": len(chunks),
        "source_files": source_files,
        "keyword_overlap": float(overlap),
    }


def build_system_prompt(query: str, context_stats: Dict[str, object]) -> str:
    num_chunks = context_stats.get("num_chunks", "unknown")
    source_files = context_stats.get("source_files", [])
    keyword_overlap = context_stats.get("keyword_overlap", "unknown")

    return f"""
You are a retrieval-augmented QA assistant. You MUST answer using ONLY the provided context chunks from user documents
(RFPs, resumes, proposals, and similar business documents). If the answer is not clearly present, say so.

Guidelines:
- Read ALL retrieved chunks before answering.
- Your answer must be on point, but you should include all important details from the context that are clearly relevant to the question, even if the answer becomes long.
- Start with a direct summary, then add additional important points as short bullets or short paragraphs.
- For exact-field questions (date, time, amount, validity, ratio, EMD, payment terms, deadlines): use a quote-first approach:
  1) Identify the single best line(s) in the context that contains the exact value.
  2) Copy the value EXACTLY (do not rewrite numbers/dates/times).
  3) If multiple different values appear, list all candidates with citations and say the documents conflict; do not guess.
- When listing projects or certifications, list all clearly mentioned items with brief one-line descriptions if available.
- Always include at least one evidence line with a quote and the chunk id like [1].

Hard restrictions:
- Do NOT invent or guess values that are not in the context.
- Do NOT use world knowledge; rely only on the retrieved chunks.
- Do NOT claim \"Not found in the indexed documents.\" if any chunk obviously contains the requested field or section.

When information is truly missing:
- Reply exactly: \"Not found in the indexed documents.\" (no extra text).

Current question: \"{query}\"
Context summary (for you, not for the user):
- number_of_chunks: {num_chunks}
- source_files: {", ".join(source_files) if isinstance(source_files, list) else source_files}
- keyword_overlap: {keyword_overlap}
""".strip()


def build_user_prompt(query: str, context: str) -> str:
    return f"""
### Question
{query}

### Retrieved Context
Use ONLY this context to answer the question. Treat it as ground truth.

{context}

### Response format (must follow exactly)
Answer:
- <start with a concise summary sentence or two, then include all other clearly important relevant details in short paragraphs or bullet points as needed. If the answer truly does not appear in the context, reply exactly: "Not found in the indexed documents.">

Evidence:
- "<short supporting quote from the most relevant chunk>" [chunk_number]
- "<optional extra quote if needed>" [chunk_number]
""".strip()


def answer_with_context(
    query: str,
    vectorstore: LangchainPinecone,
    llm: ChatGroq,
    *,
    doc_name_filter: Optional[str] = None,
    k: int = 8,
) -> Tuple[str, List[RetrievedChunk]]:
    # Slightly higher default improves on-pointness after reranking
    k = max(8, int(k or 0))
    chunks = retrieve_with_scores(query, vectorstore, doc_name_filter=doc_name_filter, final_k=k)
    if not chunks:
        return "Not found in the indexed documents.", []

    # Token-budget the context to avoid Groq 413 TPM "request too large" errors
    context = format_cited_context(chunks, max_context_tokens=3400)

    stats = build_context_stats(query, chunks)
    system_prompt = build_system_prompt(query, stats)
    user_prompt = build_user_prompt(query, context)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    resp = llm.invoke(messages)
    return resp.content, chunks


# ---------- Streamlit UI ----------


def main():
    st.set_page_config(page_title="Chat with your document", layout="wide")
    st.title("📄 Chat with your documents")

    st.sidebar.header("Index & Upload")
    debug_mode = st.sidebar.toggle("Debug mode (show retrieved context)", value=False)
    if "doc_names" not in st.session_state:
        # Load persistent catalog of document names if it exists
        if os.path.exists(DOC_CATALOG_PATH):
            try:
                with open(DOC_CATALOG_PATH, "r", encoding="utf-8") as f:
                    st.session_state["doc_names"] = json.load(f) or []
            except Exception:
                st.session_state["doc_names"] = []
        else:
            st.session_state["doc_names"] = []
    doc_options = ["All documents"] + sorted(set(st.session_state["doc_names"]))
    selected_doc = st.sidebar.selectbox(
        "Doc name filter",
        options=doc_options,
        index=0,
        help="Select a document to limit retrieval to that file.",
    )
    doc_filter = "" if selected_doc == "All documents" else selected_doc
    uploaded_files = st.sidebar.file_uploader(
        "Upload files (PDF, DOC, DOCX, images, TXT)",
        type=["pdf", "doc", "docx", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True,
    )

    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "index_initialized" not in st.session_state:
        st.session_state["index_initialized"] = False
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    index_name = DEFAULT_INDEX_NAME

    # Try to auto-connect to an existing Pinecone index on first load
    if not st.session_state["index_initialized"]:
        try:
            pc = init_pinecone(index_name=index_name)
            embeddings = get_embeddings()
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            if stats.get("total_vector_count", 0) > 0:
                st.session_state["vectorstore"] = LangchainPinecone.from_existing_index(
                    index_name=index_name,
                    embedding=embeddings,
                )
                st.sidebar.info(
                    f"Connected to existing Pinecone index '{index_name}' "
                    f"with {stats.get('total_vector_count', 0)} vectors."
                )
            st.session_state["index_initialized"] = True
        except Exception as e:
            st.sidebar.warning(f"Could not auto-connect to Pinecone index: {e}")

    with st.sidebar:
        if st.button("Process documents") and uploaded_files:
            with st.spinner("Processing and indexing documents..."):
                try:
                    # 1) Load & chunk
                    raw_docs = load_uploaded_files(uploaded_files)
                    
                    if not raw_docs:
                        st.error("No documents were loaded. Please check your files and try again.")
                        return
                    
                    chunks = chunk_docs(raw_docs)
                    
                    if not chunks:
                        st.error("No text chunks were created from the documents. The files may be empty or unreadable.")
                        return
                    # Track document names for dropdown filter (session + persistent catalog)
                    new_names = {
                        (d.metadata or {}).get("source_file") or (d.metadata or {}).get("source") or "uploaded"
                        for d in raw_docs
                    }
                    all_names = sorted(set(st.session_state.get("doc_names", [])) | new_names)
                    st.session_state["doc_names"] = all_names
                    try:
                        with open(DOC_CATALOG_PATH, "w", encoding="utf-8") as f:
                            json.dump(all_names, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                    # 2) Init Pinecone & embeddings
                    init_pinecone(index_name=index_name)
                    embeddings = get_embeddings()

                    # 3) Get current index stats before adding
                    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                    index = pc.Index(index_name)
                    stats_before = index.describe_index_stats()
                    count_before = stats_before.get("total_vector_count", 0)

                    # 4) Connect vectorstore and add docs with deterministic IDs (dedup)
                    vectorstore = LangchainPinecone.from_existing_index(
                        index_name=index_name, embedding=embeddings
                    )
                    for ci, d in enumerate(chunks):
                        d.metadata["chunk_index"] = ci
                    attempted = add_documents_deduped(
                        vectorstore,
                        chunks,
                    )

                    # 5) Verify documents were actually added
                    stats_after = index.describe_index_stats()
                    count_after = stats_after.get("total_vector_count", 0)
                    added_count = count_after - count_before

                    st.session_state["vectorstore"] = vectorstore
                    
                    if added_count > 0:
                        st.success(
                            f"✅ Successfully indexed {added_count} new chunks into namespace '{DEFAULT_NAMESPACE}'. "
                            f"Total vectors in index: {count_after}"
                        )
                    else:
                        st.warning(
                            f"⚠️ No new documents were added. Index already contains {count_after} vectors. "
                            f"This upload likely deduped (attempted {attempted} chunks) or extracted empty text."
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.exception(e)

    # Chat area
    st.subheader("Chat")

    # Show chat history
    for msg in st.session_state["messages"]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        if st.session_state["vectorstore"] is None:
            st.warning("Please upload and process documents first.")
            return

        llm = get_llm()

        # Add user message to history
        user_msg = HumanMessage(content=user_input)
        st.session_state["messages"].append(user_msg)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, retrieved = answer_with_context(
                    query=user_input,
                    vectorstore=st.session_state["vectorstore"],
                    llm=llm,
                    doc_name_filter=doc_filter or None,
                    k=8,
                )
                st.markdown(answer)
                if debug_mode and retrieved:
                    with st.expander("Debug: retrieved chunks"):
                        for i, ch in enumerate(retrieved, 1):
                            md = getattr(ch.doc, "metadata", {}) or {}
                            st.caption(
                                f"[{i}] final={ch.final_score:.4f} dense={ch.dense_score:.4f} kw={ch.kw_score:.4f} "
                                f"source={md.get('source_file') or md.get('source')} page={md.get('page')}"
                            )
                            st.text((getattr(ch.doc, "page_content", "") or "")[:1200])

        # Save assistant reply
        st.session_state["messages"].append(AIMessage(content=answer))


if __name__ == "__main__":
    main()

