import os
import json
import time
from uuid import uuid4

import requests
import streamlit as st
from dotenv import load_dotenv

from src.llm import (
    stream_response,
    get_available_models,
    generate_document_questions,
    rewrite_text,
    summarize_document,
)
from src.pdf_utils import extract_text_from_pdf
from src.rag_utils import get_top_relevant_chunks_with_scores, split_text_into_chunks
from src.embeddings_utils import embed_chunks, semantic_search, DEFAULT_EMBEDDING_MODEL
from src.storage import (
    generate_conversation_id,
    save_conversation,
    load_conversation,
    list_conversations,
    delete_conversation,
    set_conversation_favorite,
)

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def check_ollama_connection():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        models = []
        for item in data.get("models", []):
            if "name" in item:
                models.append(item["name"])

        return True, models
    except Exception as e:
        return False, str(e)


def export_chat_as_markdown(messages):
    lines = ["# Conversation avec le chatbot", ""]

    for message in messages:
        role = "Utilisateur" if message["role"] == "user" else "Assistant"
        lines.append(f"## {role}")
        lines.append(message["content"])
        lines.append("")

    return "\n".join(lines)


def export_chat_as_txt(messages):
    lines = ["Conversation avec le chatbot", "=" * 30, ""]

    for message in messages:
        role = "Utilisateur" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}:")
        lines.append(message["content"])
        lines.append("")

    return "\n".join(lines)


def export_chat_as_json(messages, metadata=None):
    payload = {
        "metadata": metadata or {},
        "messages": messages,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def get_last_user_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content", "").strip():
            return msg["content"]
    return None


def get_last_assistant_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content", "").strip():
            return msg["content"]
    return None


def get_document_by_id(documents, document_id):
    for doc in documents:
        if doc["id"] == document_id:
            return doc
    return None


def get_active_document():
    if not st.session_state.documents:
        return None

    active_id = st.session_state.active_document_id
    doc = get_document_by_id(st.session_state.documents, active_id)

    if doc is None:
        doc = st.session_state.documents[0]
        st.session_state.active_document_id = doc["id"]

    return doc


def add_uploaded_documents(uploaded_files):
    for uploaded_pdf in uploaded_files:
        try:
            extracted_text = extract_text_from_pdf(uploaded_pdf)
            chunks = split_text_into_chunks(extracted_text)

            summary = summarize_document(
                extracted_text,
                model_name=st.session_state.selected_model
            )

            questions = generate_document_questions(
                extracted_text,
                model_name=st.session_state.selected_model
            )

            embeddings = []
            if chunks:
                embeddings = embed_chunks(
                    chunks,
                    model_name=EMBEDDING_MODEL
                )

            new_doc = {
                "id": str(uuid4()),
                "name": uploaded_pdf.name,
                "text": extracted_text,
                "chunks": chunks,
                "summary": summary,
                "questions": questions,
                "embeddings": embeddings,
            }

            st.session_state.documents.append(new_doc)
            st.session_state.active_document_id = new_doc["id"]

        except Exception as e:
            st.error(f"Erreur lors de la lecture du PDF {uploaded_pdf.name} : {e}")


def remove_active_document():
    active_doc = get_active_document()
    if not active_doc:
        return

    st.session_state.documents = [
        doc for doc in st.session_state.documents
        if doc["id"] != active_doc["id"]
    ]

    if st.session_state.documents:
        st.session_state.active_document_id = st.session_state.documents[0]["id"]
    else:
        st.session_state.active_document_id = None


def build_multi_doc_context(query: str, documents: list[dict], rag_mode: str = "semantic", top_k_per_doc: int = 2, global_top_k: int = 5):
    all_hits = []

    for doc in documents:
        if not doc["chunks"]:
            continue

        if rag_mode == "semantic" and len(doc.get("embeddings", [])) > 0:
            hits = semantic_search(
                query,
                doc["chunks"],
                doc["embeddings"],
                model_name=EMBEDDING_MODEL,
                top_k=top_k_per_doc
            )
        else:
            hits = get_top_relevant_chunks_with_scores(
                query,
                doc["chunks"],
                top_k=top_k_per_doc
            )

        for hit in hits:
            all_hits.append(
                {
                    "document_name": doc["name"],
                    "document_id": doc["id"],
                    "chunk_id": hit["chunk_id"],
                    "score": hit["score"],
                    "text": hit["text"],
                }
            )

    all_hits.sort(key=lambda item: item["score"], reverse=True)
    selected_hits = all_hits[:global_top_k]

    context_parts = []
    for i, item in enumerate(selected_hits, start=1):
        context_parts.append(
            f"[Source {i}] Document: {item['document_name']} | Chunk: {item['chunk_id']} | Score: {item['score']:.4f}\n"
            f"{item['text']}"
        )

    document_context = "\n\n---\n\n".join(context_parts)
    return document_context, selected_hits


st.set_page_config(
    page_title="Chatbot IA avec Ollama",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }

    .hero-box {
        padding: 1.4rem 1.5rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(112, 130, 255, 0.10), rgba(170, 130, 255, 0.08));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.6rem;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.88;
    }

    .card {
        padding: 1rem 1rem 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 185px;
    }

    .card h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.35rem;
    }

    .small-muted {
        font-size: 0.96rem;
        opacity: 0.82;
        margin-bottom: 0.4rem;
    }

    .soft-box {
        border-radius: 16px;
        padding: 0.8rem 1rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 0.75rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 650;
        margin-bottom: 0.4rem;
    }

    div[data-testid="stMetric"] {
        background: transparent;
        border: none;
        padding: 0;
    }

    div[data-testid="stExpander"] details {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
    }

    div[data-testid="stChatMessage"] {
        border-radius: 16px;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ollama_status" not in st.session_state:
    st.session_state.ollama_status = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_OLLAMA_MODEL

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "regenerate_prompt" not in st.session_state:
    st.session_state.regenerate_prompt = None

if "rewrite_mode" not in st.session_state:
    st.session_state.rewrite_mode = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None

if "document_scope" not in st.session_state:
    st.session_state.document_scope = "active"

if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "semantic"

if "strict_document_mode" not in st.session_state:
    st.session_state.strict_document_mode = False

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = generate_conversation_id()

if "history_search_query" not in st.session_state:
    st.session_state.history_search_query = ""

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = None

available_models = get_available_models()
saved_conversations = list_conversations(st.session_state.history_search_query)

if st.session_state.selected_model not in available_models and len(available_models) > 0:
    if DEFAULT_OLLAMA_MODEL in available_models:
        st.session_state.selected_model = DEFAULT_OLLAMA_MODEL
    else:
        st.session_state.selected_model = available_models[0]

current_title = "Conversation sans titre"
if st.session_state.messages:
    first_user_message = next(
        (
            m["content"]
            for m in st.session_state.messages
            if m.get("role") == "user" and m.get("content")
        ),
        None,
    )
    if first_user_message:
        current_title = first_user_message[:60]

current_favorite = False
current_data = load_conversation(st.session_state.conversation_id)
if current_data:
    current_favorite = current_data.get("metadata", {}).get("favorite", False)

message_count = len(st.session_state.messages)
active_document = get_active_document()

export_metadata = {
    "conversation_id": st.session_state.conversation_id,
    "title": current_title,
    "model": st.session_state.selected_model,
    "embedding_model": EMBEDDING_MODEL,
    "rag_mode": st.session_state.rag_mode,
    "active_document_name": active_document["name"] if active_document else None,
    "document_names": [doc["name"] for doc in st.session_state.documents],
    "document_scope": st.session_state.document_scope,
    "strict_document_mode": st.session_state.strict_document_mode,
    "favorite": current_favorite,
    "message_count": message_count,
    "last_response_time_seconds": st.session_state.last_response_time,
}

with st.sidebar:
    st.title("⚙️ Configuration")

    if len(available_models) > 0:
        selected_model = st.selectbox(
            "Choisis un modèle",
            options=available_models,
            index=available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models else 0
        )
        st.session_state.selected_model = selected_model
    else:
        st.warning("Aucun modèle détecté dans Ollama.")
        st.session_state.selected_model = DEFAULT_OLLAMA_MODEL

    st.session_state.rag_mode = st.radio(
        "Mode RAG",
        options=["semantic", "lexical"],
        format_func=lambda x: "Sémantique (embeddings)" if x == "semantic" else "Lexical (mots-clés)",
        index=0 if st.session_state.rag_mode == "semantic" else 1
    )

    st.markdown(f"**Modèle actif :** `{st.session_state.selected_model}`")
    st.markdown(f"**Embeddings :** `{EMBEDDING_MODEL}`")
    st.markdown(f"**Conversation :** `{st.session_state.conversation_id}`")
    st.markdown(f"**Titre :** {current_title}")
    st.markdown(f"**Favori :** {'⭐ Oui' if current_favorite else '—'}")
    st.markdown(f"**Messages :** {message_count}")

    if st.session_state.last_response_time is not None:
        st.markdown(f"**Dernière réponse :** {st.session_state.last_response_time:.2f} sec")
    else:
        st.markdown("**Dernière réponse :** —")

    st.divider()

    if st.button("🔍 Tester la connexion à Ollama"):
        is_connected, result = check_ollama_connection()
        st.session_state.ollama_status = (is_connected, result)

    if st.session_state.ollama_status is not None:
        is_connected, result = st.session_state.ollama_status
        if is_connected:
            st.success("Connexion à Ollama réussie ✅")
            if len(result) > 0:
                st.markdown("**Modèles détectés :**")
                for model_name in result:
                    st.markdown(f"- `{model_name}`")
        else:
            st.error("Connexion à Ollama impossible ❌")
            st.caption(str(result))

    st.divider()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        if st.button("🔁 Régénérer la dernière réponse"):
            last_user_prompt = get_last_user_message(st.session_state.messages)
            if last_user_prompt:
                st.session_state.messages.pop()
                st.session_state.regenerate_prompt = last_user_prompt
                st.rerun()

        rewrite_col1, rewrite_col2, rewrite_col3 = st.columns(3)

        with rewrite_col1:
            if st.button("✂️ Plus court"):
                st.session_state.rewrite_mode = "shorter"
                st.rerun()

        with rewrite_col2:
            if st.button("💼 Plus pro"):
                st.session_state.rewrite_mode = "professional"
                st.rerun()

        with rewrite_col3:
            if st.button("🪶 Plus simple"):
                st.session_state.rewrite_mode = "simpler"
                st.rerun()

    st.divider()
    st.markdown("### 📚 Documents")

    if st.session_state.documents:
        document_options = {doc["id"]: doc["name"] for doc in st.session_state.documents}

        selected_doc_id = st.selectbox(
            "Document actif",
            options=list(document_options.keys()),
            format_func=lambda doc_id: document_options[doc_id],
            index=list(document_options.keys()).index(st.session_state.active_document_id)
            if st.session_state.active_document_id in document_options else 0
        )
        st.session_state.active_document_id = selected_doc_id
        active_document = get_active_document()

        st.session_state.document_scope = st.radio(
            "Portée de recherche",
            options=["active", "all"],
            format_func=lambda x: "Document actif" if x == "active" else "Tous les documents",
            index=0 if st.session_state.document_scope == "active" else 1
        )

        st.session_state.strict_document_mode = st.checkbox(
            "Répondre uniquement à partir du périmètre choisi",
            value=st.session_state.strict_document_mode
        )

        st.caption(f"{len(st.session_state.documents)} document(s) chargé(s)")
        st.caption(f"Chunks : {len(active_document['chunks'])}")

        with st.expander("Voir un aperçu du document actif"):
            preview = active_document["text"][:1800]
            if preview:
                st.text(preview)
            else:
                st.warning("Aucun texte n'a pu être extrait de ce document.")

        if st.button("🗑️ Retirer le document actif"):
            remove_active_document()
            st.rerun()
    else:
        st.caption("Aucun document chargé.")

    st.divider()
    st.markdown("### 💾 Historique")

    st.session_state.history_search_query = st.text_input(
        "Rechercher",
        value=st.session_state.history_search_query,
        placeholder="Tape un mot-clé..."
    )

    saved_conversations = list_conversations(st.session_state.history_search_query)

    if saved_conversations:
        conversation_labels = {
            c["conversation_id"]: f'{"⭐ " if c["favorite"] else ""}{c["title"]} ({c["message_count"]} messages)'
            for c in saved_conversations
        }

        selected_saved_conversation = st.selectbox(
            "Choisir une conversation",
            options=list(conversation_labels.keys()),
            format_func=lambda cid: conversation_labels[cid]
        )

        if st.button("📂 Charger cette conversation"):
            data = load_conversation(selected_saved_conversation)
            if data:
                st.session_state.messages = data.get("messages", [])
                st.session_state.conversation_id = data.get("conversation_id", generate_conversation_id())
                st.success("Conversation chargée ✅")
                st.rerun()

        fav_col1, fav_col2 = st.columns(2)

        with fav_col1:
            if st.button("⭐ Épingler / Retirer"):
                data = load_conversation(selected_saved_conversation)
                if data:
                    current_state = data.get("metadata", {}).get("favorite", False)
                    updated = set_conversation_favorite(selected_saved_conversation, not current_state)
                    if updated:
                        st.success("Favori mis à jour ✅")
                        st.rerun()
                    else:
                        st.error("Impossible de mettre à jour le favori.")

        with fav_col2:
            if st.button("🗑️ Supprimer cette conversation"):
                deleted = delete_conversation(selected_saved_conversation)
                if deleted:
                    if st.session_state.conversation_id == selected_saved_conversation:
                        st.session_state.messages = []
                        st.session_state.conversation_id = generate_conversation_id()
                    st.success("Conversation supprimée ✅")
                    st.rerun()
                else:
                    st.error("Impossible de supprimer cette conversation.")
    else:
        st.caption("Aucune conversation trouvée.")

    st.divider()

    if st.button("🧹 Nouvelle conversation"):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.session_state.regenerate_prompt = None
        st.session_state.rewrite_mode = None
        st.session_state.conversation_id = generate_conversation_id()
        st.session_state.documents = []
        st.session_state.active_document_id = None
        st.session_state.document_scope = "active"
        st.session_state.rag_mode = "semantic"
        st.session_state.strict_document_mode = False
        st.session_state.last_response_time = None
        st.rerun()

    if len(st.session_state.messages) > 0:
        markdown_export = export_chat_as_markdown(st.session_state.messages)
        txt_export = export_chat_as_txt(st.session_state.messages)
        json_export = export_chat_as_json(st.session_state.messages, export_metadata)

        st.download_button(
            label="💾 Markdown",
            data=markdown_export,
            file_name="conversation_chatbot.md",
            mime="text/markdown"
        )

        st.download_button(
            label="📄 TXT",
            data=txt_export,
            file_name="conversation_chatbot.txt",
            mime="text/plain"
        )

        st.download_button(
            label="🧩 JSON",
            data=json_export,
            file_name="conversation_chatbot.json",
            mime="application/json"
        )

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🤖 Chatbot IA local avec Ollama</div>
        <div class="hero-subtitle">
            Une app Streamlit gratuite avec modèle open source en local.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

status_parts = [
    f"Modèle : {st.session_state.selected_model}",
    f"RAG : {'Sémantique' if st.session_state.rag_mode == 'semantic' else 'Lexical'}",
    f"Messages : {message_count}",
]

if st.session_state.last_response_time is not None:
    status_parts.append(f"Dernière réponse : {st.session_state.last_response_time:.2f} sec")

if st.session_state.documents:
    if st.session_state.document_scope == "all":
        status_parts.append(f"Documents : {len(st.session_state.documents)} (tous)")
    else:
        active_document = get_active_document()
        if active_document:
            status_parts.append(f"Document actif : {active_document['name']}")

st.caption(" · ".join(status_parts))

active_document = get_active_document()

if active_document:
    st.markdown(
        f"""
        <div class="soft-box">
            <div class="section-title">📘 Mode document actif</div>
            <div>{active_document['name']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if active_document["summary"]:
        with st.expander("📝 Résumé automatique du document actif", expanded=True):
            st.markdown(active_document["summary"])

    if active_document["questions"]:
        st.markdown("### 💡 Questions suggérées")

        qcol1, qcol2 = st.columns(2)

        for idx, question in enumerate(active_document["questions"]):
            target_col = qcol1 if idx % 2 == 0 else qcol2
            with target_col:
                if st.button(question, key=f"doc_question_{idx}"):
                    st.session_state.pending_prompt = question
                    st.rerun()

if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="small-muted">
            Pose une question à ton assistant local. Tu peux discuter librement,
            analyser un ou plusieurs PDF, demander un résumé ou comparer des documents.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>📘 Expliquer</h3>
                <p>Idéal pour comprendre un concept ou une notion technique.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.code("Explique-moi les bases du machine learning simplement", language=None)
        if st.button("Utiliser cet exemple", key="example_1"):
            st.session_state.pending_prompt = "Explique-moi les bases du machine learning simplement"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>✍️ Rédiger</h3>
                <p>Pratique pour écrire un message, un email ou reformuler un texte.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.code("Aide-moi à écrire un email professionnel pour demander un stage", language=None)
        if st.button("Utiliser cet exemple", key="example_2"):
            st.session_state.pending_prompt = "Aide-moi à écrire un email professionnel pour demander un stage"
            st.rerun()

    with col3:
        st.markdown(
            """
            <div class="card">
                <h3>🧠 Comparer / apprendre</h3>
                <p>Utile pour obtenir un plan, un résumé croisé ou une synthèse documentaire.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.code("Compare ces documents et donne-moi les différences clés", language=None)
        if st.button("Utiliser cet exemple", key="example_3"):
            st.session_state.pending_prompt = "Compare ces documents et donne-moi les différences clés"
            st.rerun()

    st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

submission = st.chat_input(
    "Pose votre question ici...",
    accept_file="multiple",
    file_type=["pdf"],
)

final_prompt = None
regenerated = False

if submission is not None:
    uploaded_files = submission.files if hasattr(submission, "files") else []
    user_text = submission.text if hasattr(submission, "text") else ""

    if uploaded_files:
        add_uploaded_documents(uploaded_files)

    if user_text.strip():
        final_prompt = user_text.strip()

if final_prompt is None and st.session_state.pending_prompt is not None:
    final_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if final_prompt is None and st.session_state.regenerate_prompt is not None:
    final_prompt = st.session_state.regenerate_prompt
    st.session_state.regenerate_prompt = None
    regenerated = True

if st.session_state.rewrite_mode is not None:
    last_assistant_text = get_last_assistant_message(st.session_state.messages)
    if last_assistant_text:
        rewritten = rewrite_text(
            last_assistant_text,
            rewrite_mode=st.session_state.rewrite_mode,
            model_name=st.session_state.selected_model
        )

        label_map = {
            "shorter": "Réécriture plus courte",
            "professional": "Réécriture plus professionnelle",
            "simpler": "Réécriture plus simple",
        }
        rewrite_label = label_map.get(st.session_state.rewrite_mode, "Réécriture")

        if rewritten:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"**{rewrite_label}**\n\n{rewritten}"
                }
            )

            save_conversation(
                st.session_state.conversation_id,
                st.session_state.messages,
                metadata={
                    "model": st.session_state.selected_model,
                    "embedding_model": EMBEDDING_MODEL,
                    "rag_mode": st.session_state.rag_mode,
                    "active_document_name": active_document["name"] if active_document else None,
                    "document_names": [doc["name"] for doc in st.session_state.documents],
                    "document_scope": st.session_state.document_scope,
                    "strict_document_mode": st.session_state.strict_document_mode,
                    "title": current_title,
                    "favorite": current_favorite,
                },
            )

    st.session_state.rewrite_mode = None
    st.rerun()

if final_prompt:
    if not regenerated:
        st.session_state.messages.append(
            {"role": "user", "content": final_prompt}
        )

        with st.chat_message("user"):
            st.markdown(final_prompt)

    document_context = ""
    source_items = []

    if st.session_state.documents:
        if st.session_state.document_scope == "all":
            document_context, source_items = build_multi_doc_context(
                final_prompt,
                st.session_state.documents,
                rag_mode=st.session_state.rag_mode,
                top_k_per_doc=2,
                global_top_k=5
            )
        else:
            active_document = get_active_document()
            if active_document and active_document["chunks"]:
                if st.session_state.rag_mode == "semantic" and len(active_document.get("embeddings", [])) > 0:
                    hits = semantic_search(
                        final_prompt,
                        active_document["chunks"],
                        active_document["embeddings"],
                        model_name=EMBEDDING_MODEL,
                        top_k=3
                    )
                else:
                    hits = get_top_relevant_chunks_with_scores(
                        final_prompt,
                        active_document["chunks"],
                        top_k=3
                    )

                source_items = [
                    {
                        "document_name": active_document["name"],
                        "document_id": active_document["id"],
                        "chunk_id": item["chunk_id"],
                        "score": item["score"],
                        "text": item["text"],
                    }
                    for item in hits
                ]

                context_parts = []
                for i, item in enumerate(source_items, start=1):
                    context_parts.append(
                        f"[Source {i}] Document: {item['document_name']} | Chunk: {item['chunk_id']} | Score: {item['score']:.4f}\n"
                        f"{item['text']}"
                    )
                document_context = "\n\n---\n\n".join(context_parts)

    with st.chat_message("assistant"):
        try:
            start_time = time.perf_counter()

            assistant_text = st.write_stream(
                stream_response(
                    st.session_state.messages,
                    model_name=st.session_state.selected_model,
                    document_context=document_context,
                    strict_document_mode=st.session_state.strict_document_mode
                )
            )

            st.session_state.last_response_time = time.perf_counter() - start_time

            if source_items:
                with st.expander("📚 Sources utilisées pour répondre"):
                    for i, item in enumerate(source_items, start=1):
                        st.markdown(
                            f"**Source {i}** — `{item['document_name']}` — score `{item['score']:.4f}` — chunk #{item['chunk_id']}"
                        )
                        st.text(item["text"][:1200])

            st.caption(
                f"⏱️ Réponse générée en {st.session_state.last_response_time:.2f} sec avec {st.session_state.selected_model}"
            )

        except Exception as e:
            assistant_text = f"Erreur : impossible de contacter Ollama. Détail : {e}"
            st.error(assistant_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text}
    )

    save_conversation(
        st.session_state.conversation_id,
        st.session_state.messages,
        metadata={
            "model": st.session_state.selected_model,
            "embedding_model": EMBEDDING_MODEL,
            "rag_mode": st.session_state.rag_mode,
            "active_document_name": active_document["name"] if active_document else None,
            "document_names": [doc["name"] for doc in st.session_state.documents],
            "document_scope": st.session_state.document_scope,
            "strict_document_mode": st.session_state.strict_document_mode,
            "title": final_prompt[:60] if len(st.session_state.messages) <= 2 else current_title,
            "favorite": current_favorite,
        },
    )