from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .caching import get_spacy_nlp


def chunk_docs(docs, chunk_size: int = 900, chunk_overlap: int = 200):
    """
    Sentence-based chunking:
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

    max_chars = 4000
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
                chunk_text = " ".join(current_sents).strip()
                if chunk_text:
                    out.append(Document(page_content=chunk_text, metadata=dict(getattr(d, "metadata", {}) or {})))

                overlap = current_sents[-overlap_sents:] if len(current_sents) > overlap_sents else current_sents
                current_sents = overlap + [sent]
                current_len = len(" ".join(current_sents))
            else:
                current_sents.append(sent)
                current_len += (1 if current_sents else 0) + s_len

        if current_sents:
            chunk_text = " ".join(current_sents).strip()
            if chunk_text:
                out.append(Document(page_content=chunk_text, metadata=dict(getattr(d, "metadata", {}) or {})))

    return out

