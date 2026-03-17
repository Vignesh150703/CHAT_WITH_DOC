from typing import List

from langchain_pinecone import Pinecone as LangchainPinecone

from .config import DEFAULT_NAMESPACE
from .text_utils import sha256_bytes


def stable_chunk_id(namespace: str, source_file: str, page: int, chunk_index: int, text: str) -> str:
    base = f"{namespace}|{source_file}|{page}|{chunk_index}|{text}".encode("utf-8", errors="ignore")
    return sha256_bytes(base)


def add_documents_deduped(vectorstore: LangchainPinecone, docs) -> int:
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

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids, namespace=DEFAULT_NAMESPACE)
    return len(ids)

