import os
import json
import requests
from dotenv import load_dotenv

from src.prompts import SYSTEM_PROMPT

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3")


def build_messages(
    chat_history: list[dict],
    document_context: str = "",
    strict_document_mode: bool = False
) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    if document_context.strip():
        if strict_document_mode:
            document_instruction = (
                "Tu disposes d'extraits pertinents d'un document fourni par l'utilisateur. "
                "Tu dois répondre uniquement à partir de ces extraits. "
                "Si l'information n'est pas présente dans ces extraits, dis clairement que tu ne peux pas répondre "
                "à partir du document fourni. N'invente rien."
            )
        else:
            document_instruction = (
                "Tu disposes d'extraits pertinents d'un document fourni par l'utilisateur. "
                "Réponds en priorité à partir de ces extraits. "
                "Si l'information n'est pas présente, dis-le honnêtement."
            )

        messages.append(
            {
                "role": "system",
                "content": (
                    f"{document_instruction}\n\n"
                    f"Extraits du document :\n{document_context}"
                )
            }
        )

    for msg in chat_history:
        messages.append(
            {
                "role": msg["role"],
                "content": msg["content"]
            }
        )

    return messages


def get_available_models() -> list[str]:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        models = []
        for item in data.get("models", []):
            if "name" in item:
                models.append(item["name"])

        return models
    except Exception:
        return []


def stream_response(
    chat_history: list[dict],
    model_name: str | None = None,
    document_context: str = "",
    strict_document_mode: bool = False
):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    messages = build_messages(
        chat_history,
        document_context=document_context,
        strict_document_mode=strict_document_mode
    )

    selected_model = model_name or DEFAULT_OLLAMA_MODEL

    payload = {
        "model": selected_model,
        "messages": messages,
        "stream": True
    }

    with requests.post(url, json=payload, stream=True, timeout=120) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))

                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]


def generate_document_questions(document_text: str, model_name: str | None = None) -> list[str]:
    if not document_text.strip():
        return []

    selected_model = model_name or DEFAULT_OLLAMA_MODEL
    url = f"{OLLAMA_BASE_URL}/api/chat"

    prompt = (
        "Tu reçois un document. Génère exactement 5 questions utiles, claires et variées "
        "qu'un utilisateur pourrait poser sur ce document.\n"
        "Les questions doivent être en français.\n"
        "Elles doivent aider à comprendre le document.\n"
        "Réponds uniquement sous forme de liste, une question par ligne, sans commentaire.\n\n"
        f"Document :\n{document_text[:6000]}"
    )

    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "").strip()
        if not content:
            return []

        lines = [line.strip("-•1234567890. \n\t") for line in content.splitlines()]
        questions = [line for line in lines if line and len(line) > 8]

        cleaned_questions = []
        for q in questions:
            if not q.endswith("?"):
                q = q + "?"
            if q not in cleaned_questions:
                cleaned_questions.append(q)

        return cleaned_questions[:5]

    except Exception:
        return []


def rewrite_text(
    original_text: str,
    rewrite_mode: str,
    model_name: str | None = None
) -> str:
    if not original_text.strip():
        return ""

    selected_model = model_name or DEFAULT_OLLAMA_MODEL
    url = f"{OLLAMA_BASE_URL}/api/chat"

    mode_instructions = {
        "shorter": "Réécris ce texte en le rendant plus court, tout en gardant les idées essentielles.",
        "professional": "Réécris ce texte dans un ton plus professionnel, clair et soigné.",
        "simpler": "Réécris ce texte dans un style plus simple, plus facile à comprendre."
    }

    instruction = mode_instructions.get(
        rewrite_mode,
        "Réécris ce texte de façon claire."
    )

    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{instruction}\n\nTexte à réécrire :\n{original_text}"
            },
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception:
        return ""


def summarize_document(document_text: str, model_name: str | None = None) -> str:
    if not document_text.strip():
        return ""

    selected_model = model_name or DEFAULT_OLLAMA_MODEL
    url = f"{OLLAMA_BASE_URL}/api/chat"

    prompt = (
        "Tu reçois un document. Fais un résumé clair en français avec :\n"
        "1. un court paragraphe d'ensemble\n"
        "2. 5 points clés maximum\n"
        "Reste fidèle au document et n'invente rien.\n\n"
        f"Document :\n{document_text[:8000]}"
    )

    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception:
        return ""