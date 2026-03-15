import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    return SentenceTransformer(model_name)


def embed_chunks(chunks: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL):
    if not chunks:
        return []

    model = get_embedding_model(model_name)

    if hasattr(model, "encode_document"):
        vectors = model.encode_document(chunks, convert_to_numpy=True, show_progress_bar=False)
    else:
        vectors = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

    return vectors


def embed_query(query: str, model_name: str = DEFAULT_EMBEDDING_MODEL):
    model = get_embedding_model(model_name)

    if hasattr(model, "encode_query"):
        vector = model.encode_query(query, convert_to_numpy=True, show_progress_bar=False)
    else:
        vector = model.encode(query, convert_to_numpy=True, show_progress_bar=False)

    return vector


def semantic_search(
    query: str,
    chunks: list[str],
    embeddings,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 3
) -> list[dict]:
    if not query.strip() or not chunks or embeddings is None or len(embeddings) == 0:
        return []

    query_embedding = embed_query(query, model_name=model_name)
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    scored = []
    for idx, (chunk, score) in enumerate(zip(chunks, similarities), start=1):
        scored.append(
            {
                "chunk_id": idx,
                "score": float(score),
                "text": chunk,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]