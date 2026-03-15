import json
import re
from pathlib import Path
from datetime import datetime


BASE_DIR = Path("data/conversations")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def generate_conversation_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_conversation_path(conversation_id: str) -> Path:
    return BASE_DIR / f"{conversation_id}.json"


def generate_conversation_title(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if content:
                content = re.sub(r"\s+", " ", content)
                return content[:60]
    return "Conversation sans titre"


def save_conversation(conversation_id: str, messages: list[dict], metadata: dict | None = None):
    path = get_conversation_path(conversation_id)

    metadata = metadata or {}
    if "title" not in metadata or not metadata["title"]:
        metadata["title"] = generate_conversation_title(messages)
    if "favorite" not in metadata:
        metadata["favorite"] = False

    payload = {
        "conversation_id": conversation_id,
        "saved_at": datetime.now().isoformat(),
        "metadata": metadata,
        "messages": messages,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_conversation(conversation_id: str) -> dict | None:
    path = get_conversation_path(conversation_id)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_conversation(conversation_id: str) -> bool:
    path = get_conversation_path(conversation_id)

    if path.exists():
        path.unlink()
        return True

    return False


def set_conversation_favorite(conversation_id: str, is_favorite: bool) -> bool:
    data = load_conversation(conversation_id)
    if not data:
        return False

    metadata = data.get("metadata", {})
    metadata["favorite"] = is_favorite
    data["metadata"] = metadata

    path = get_conversation_path(conversation_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return True


def list_conversations(search_query: str = "") -> list[dict]:
    conversations = []
    normalized_query = search_query.strip().lower()

    for path in BASE_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = data.get("messages", [])
            metadata = data.get("metadata", {})
            preview = "Conversation vide"

            for msg in messages:
                if msg.get("role") == "user" and msg.get("content", "").strip():
                    preview = msg["content"][:60]
                    break

            title = metadata.get("title") or generate_conversation_title(messages)
            favorite = metadata.get("favorite", False)

            searchable_text = " ".join(
                [
                    title,
                    preview,
                    " ".join(msg.get("content", "") for msg in messages),
                ]
            ).lower()

            if normalized_query and normalized_query not in searchable_text:
                continue

            conversations.append(
                {
                    "conversation_id": data.get("conversation_id", path.stem),
                    "saved_at": data.get("saved_at", ""),
                    "preview": preview,
                    "title": title,
                    "message_count": len(messages),
                    "favorite": favorite,
                }
            )
        except Exception:
            continue

    conversations.sort(
        key=lambda c: (not c["favorite"], c["saved_at"]),
        reverse=False
    )
    conversations.sort(
        key=lambda c: c["saved_at"],
        reverse=True
    )
    conversations.sort(
        key=lambda c: not c["favorite"]
    )

    return conversations