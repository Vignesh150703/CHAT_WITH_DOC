from typing import Dict, List, Optional, Tuple

from langchain_pinecone import Pinecone as LangchainPinecone

from .caching import get_reranker
from .config import (
    DEFAULT_NAMESPACE,
    DENSE_K_DEFAULT,
    FINAL_K_DEFAULT,
    MAX_CHUNK_CHARS_IN_CONTEXT,
    MAX_CONTEXT_TOKENS,
    RERANK_CANDIDATES_MAX,
    RERANK_TEXT_CHARS,
    RERANK_TIEBREAK_WEIGHT,
)
from .models import RetrievedChunk
from .text_utils import count_tokens, keyword_score, sha256_bytes


def retrieve_with_scores(
    query: str,
    vectorstore: LangchainPinecone,
    *,
    doc_name_filter: Optional[str] = None,
    dense_k: int = DENSE_K_DEFAULT,
    final_k: int = FINAL_K_DEFAULT,
) -> List[RetrievedChunk]:
    dense_results: List[Tuple[object, float]] = []

    search_kwargs = {}
    if doc_name_filter:
        search_kwargs["filter"] = {"source_file": {"$eq": doc_name_filter}}

    dense_results.extend(
        vectorstore.similarity_search_with_score(
            query, k=dense_k, namespace=DEFAULT_NAMESPACE, **search_kwargs
        )
    )

    best_by_hash: Dict[str, Tuple[object, float]] = {}
    for doc, score in dense_results:
        text = (getattr(doc, "page_content", "") or "").strip()
        h = sha256_bytes(text.encode("utf-8", errors="ignore"))
        if h not in best_by_hash or float(score) > float(best_by_hash[h][1]):
            best_by_hash[h] = (doc, float(score))

    dense = list(best_by_hash.values())

    prelim: List[RetrievedChunk] = []
    for doc, score in dense:
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
            ch.final_score = (1.0 * ch.rerank_score) + (RERANK_TIEBREAK_WEIGHT * ch.final_score)

        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:final_k]
    except Exception:
        return prelim[:final_k]


def format_cited_context(
    chunks: List[RetrievedChunk],
    *,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> str:
    parts: List[str] = []
    used = 0
    for i, ch in enumerate(chunks, 1):
        doc = ch.doc
        md = getattr(doc, "metadata", {}) or {}
        src = md.get("source_file") or md.get("source") or "unknown"
        page = md.get("page", "?")
        text = (getattr(doc, "page_content", "") or "").strip()[:MAX_CHUNK_CHARS_IN_CONTEXT]
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

