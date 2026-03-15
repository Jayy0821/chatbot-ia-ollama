# Chatbot IA local avec Ollama, Streamlit et RAG sémantique

Un assistant IA local, moderne et gratuit, capable de discuter, analyser plusieurs PDF, résumer automatiquement des documents, faire de la recherche RAG et répondre avec des sources affichées dans l’interface.

---

## Démo du projet

Ce projet propose une expérience proche d’un assistant documentaire moderne, mais en **local** :

- chat IA avec **Ollama**
- interface web avec **Streamlit**
- support **multi-documents PDF**
- **résumé automatique** à l’import
- **questions suggérées** sur les documents
- **RAG lexical + sémantique**
- citations et extraits utilisés
- sauvegarde locale de conversations
- export Markdown / TXT / JSON

---
chatbot-ia-ollama/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── assets/
│   ├── screenshot-home.png
│   ├── screenshot-documents.png
│   └── screenshot-rag.png
├── data/
│   └── conversations/
└── src/
    ├── embeddings_utils.py
    ├── llm.py
    ├── pdf_utils.py
    ├── prompts.py
    ├── rag_utils.py
    └── storage.py
