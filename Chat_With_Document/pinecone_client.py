from pinecone import Pinecone, ServerlessSpec

from .config import DEFAULT_INDEX_NAME, require_env


def init_pinecone(index_name: str = DEFAULT_INDEX_NAME, dim: int = 768) -> Pinecone:
    api_key = require_env("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    existing_indexes = pc.list_indexes().names()

    # If index exists but with wrong dimension, delete and recreate it
    if index_name in existing_indexes:
        desc = pc.describe_index(index_name)
        if desc.dimension != dim:
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

