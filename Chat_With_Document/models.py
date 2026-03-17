from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    doc: object
    dense_score: float
    kw_score: float
    rerank_score: float
    final_score: float

