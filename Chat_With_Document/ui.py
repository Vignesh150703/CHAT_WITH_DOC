import json
import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone

from .caching import get_embeddings, get_llm
from .chunking import chunk_docs
from .config import (
    DEFAULT_INDEX_NAME,
    DEFAULT_NAMESPACE,
    DOC_CATALOG_PATH,
    require_env,
)
from .indexing import add_documents_deduped
from .loaders import load_uploaded_files
from .pinecone_client import init_pinecone
from .qa import answer_with_context


def _load_doc_catalog() -> list:
    if os.path.exists(DOC_CATALOG_PATH):
        try:
            with open(DOC_CATALOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or []
        except Exception:
            return []
    return []


def _save_doc_catalog(names: list) -> None:
    try:
        with open(DOC_CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(names, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main():
    st.set_page_config(page_title="Chat with your document", layout="wide")
    st.title("📄 Chat with your documents")

    st.sidebar.header("Index & Upload")
    debug_mode = st.sidebar.toggle("Debug mode (show retrieved context)", value=False)

    if "doc_names" not in st.session_state:
        st.session_state["doc_names"] = _load_doc_catalog()

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
                    raw_docs = load_uploaded_files(uploaded_files)
                    if not raw_docs:
                        st.error("No documents were loaded. Please check your files and try again.")
                        return

                    chunks = chunk_docs(raw_docs)
                    if not chunks:
                        st.error("No text chunks were created from the documents. The files may be empty or unreadable.")
                        return

                    new_names = {
                        (d.metadata or {}).get("source_file") or (d.metadata or {}).get("source") or "uploaded"
                        for d in raw_docs
                    }
                    all_names = sorted(set(st.session_state.get("doc_names", [])) | new_names)
                    st.session_state["doc_names"] = all_names
                    _save_doc_catalog(all_names)

                    init_pinecone(index_name=index_name)
                    embeddings = get_embeddings()

                    pc = Pinecone(api_key=require_env("PINECONE_API_KEY"))
                    index = pc.Index(index_name)
                    stats_before = index.describe_index_stats()
                    count_before = stats_before.get("total_vector_count", 0)

                    vectorstore = LangchainPinecone.from_existing_index(
                        index_name=index_name, embedding=embeddings
                    )
                    for ci, d in enumerate(chunks):
                        d.metadata["chunk_index"] = ci

                    attempted = add_documents_deduped(vectorstore, chunks)

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

    st.subheader("Chat")

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
        st.session_state["messages"].append(HumanMessage(content=user_input))
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

        st.session_state["messages"].append(AIMessage(content=answer))

