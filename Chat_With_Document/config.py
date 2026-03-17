import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_INDEX_NAME = "pinecone"
DEFAULT_NAMESPACE = "global"
# Keep the doc-catalog next to this package so it works regardless of CWD.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_CATALOG_PATH = os.path.join(PACKAGE_DIR, "doc_catalog.json")

# Retrieval / reranking defaults (centralized to avoid magic numbers)
DENSE_K_DEFAULT = 60
FINAL_K_DEFAULT = 10
RERANK_CANDIDATES_MAX = 50
RERANK_TEXT_CHARS = 1800
RERANK_TIEBREAK_WEIGHT = 0.01

# LLM context budgeting (keeps Groq requests under limits)
MAX_CONTEXT_TOKENS = 3400
MAX_CHUNK_CHARS_IN_CONTEXT = 3500


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ValueError(f"{name} is not set. Add it to your .env file.")
    return v

