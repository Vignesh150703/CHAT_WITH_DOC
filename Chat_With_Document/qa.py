from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_pinecone import Pinecone as LangchainPinecone

from .prompts import build_system_prompt, build_user_prompt
from .retrieval import build_context_stats, format_cited_context, retrieve_with_scores
from .models import RetrievedChunk


def answer_with_context(
    query: str,
    vectorstore: LangchainPinecone,
    llm: ChatGroq,
    *,
    doc_name_filter: Optional[str] = None,
    k: int = 8,
) -> Tuple[str, List[RetrievedChunk]]:
    k = max(8, int(k or 0))
    chunks = retrieve_with_scores(query, vectorstore, doc_name_filter=doc_name_filter, final_k=k)
    if not chunks:
        return "Not found in the indexed documents.", []

    context = format_cited_context(chunks)
    stats = build_context_stats(query, chunks)
    system_prompt = build_system_prompt(query, stats)
    user_prompt = build_user_prompt(query, context)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    resp = llm.invoke(messages)
    return resp.content, chunks

