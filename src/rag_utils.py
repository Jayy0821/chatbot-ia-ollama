import re
from collections import Counter


def split_text_into_chunks(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - overlap

    return chunks


def normalize_text(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"\b[a-zàâäéèêëîïôöùûüç0-9'-]+\b", text)
    return words


def score_chunk(query: str, chunk: str) -> int:
    query_words = normalize_text(query)
    chunk_words = normalize_text(chunk)

    if not query_words or not chunk_words:
        return 0

    query_counter = Counter(query_words)
    chunk_counter = Counter(chunk_words)

    score = 0
    for word, count in query_counter.items():
        score += min(count, chunk_counter.get(word, 0))

    return score


def get_top_relevant_chunks_with_scores(query: str, chunks: list[str], top_k: int = 3) -> list[dict]:
    scored_chunks = []

    for idx, chunk in enumerate(chunks, start=1):
        score = score_chunk(query, chunk)
        scored_chunks.append(
            {
                "chunk_id": idx,
                "score": score,
                "text": chunk,
            }
        )

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    top_chunks = [item for item in scored_chunks[:top_k] if item["score"] > 0]

    if not top_chunks:
        return scored_chunks[:top_k]

    return top_chunks


def get_top_relevant_chunks(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    return [item["text"] for item in get_top_relevant_chunks_with_scores(query, chunks, top_k=top_k)]