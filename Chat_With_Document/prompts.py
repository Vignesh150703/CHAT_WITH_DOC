from typing import Dict


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
- Do NOT claim "Not found in the indexed documents." if any chunk obviously contains the requested field or section.

When information is truly missing:
- Reply exactly: "Not found in the indexed documents." (no extra text).

Current question: "{query}"
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

