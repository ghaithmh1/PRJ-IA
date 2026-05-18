


import os
import re

import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Parametres RAG ──────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
TOP_K           = 6
RERANK_KEEP     = 3
MAX_HISTORY     = 4

# ── Etat global ─────────────────────────────────────────────────
vectorstore_global  = None
conversation_memory = []

# ── Chargement & indexation ─────────────────────────────────────

def load_and_index_pdfs(pdf_files):
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf.name)
        pages  = loader.load()
        for page in pages:
            page.metadata["source"] = os.path.basename(pdf.name)
        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks     = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs         = FAISS.from_documents(chunks, embeddings)
    return vs, len(documents), len(chunks)

# ── Re-ranking  ────────────────────────────────

def rerank_docs(docs, question, keep=RERANK_KEEP):
    keywords = set(re.findall(r'\w+', question.lower()))
    scored = []
    for doc in docs:
        words = set(re.findall(r'\w+', doc.page_content.lower()))
        score = len(keywords & words)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:keep]]


# ── Memoire de conversation ──────────────────────────────────────

def build_history_context(memory, max_turns=MAX_HISTORY):
    if not memory:
        return ""
    recent = memory[-max_turns:]
    lines  = ["Historique recent de la conversation :"]
    for i, (q, a) in enumerate(recent, 1):
        lines.append(f"[Tour {i}] Etudiant : {q}")
        lines.append(f"[Tour {i}] Assistant : {a[:250]}...")
    return "\n".join(lines)

# ── Formatage contexte ───────────────────────────────────────────

def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "inconnu")
        page   = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page += 1
        parts.append(f"[Extrait {i} | {source} | page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)

def get_sources(docs):
    seen, sources = set(), []
    for doc in docs:
        source = doc.metadata.get("source", "inconnu")
        page   = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page += 1
        key = f"{source} (p.{page})"
        if key not in seen:
            seen.add(key)
            sources.append(key)
    return sources

# ── Prompt templates ────────────────────────────────────────────

PROMPTS = {
    "question": PromptTemplate.from_template(
        "Tu es un assistant pedagogique specialise en IA generative.\n"
        "Reponds UNIQUEMENT a partir du contexte fourni. Si l'information est absente, dis-le clairement.\n"
        "Reponds toujours en francais, de facon claire et structuree.\n\n"
        "{history}\n\n"
        "Contexte :\n{context}\n\n"
        "Question : {question}\n\nReponse :"
    ),
    "resume": PromptTemplate.from_template(
        "Tu es expert en synthese de cours. Redige un resume structure en francais.\n"
        "Format : titre, points numerotes, conclusion. Max 250 mots.\n\n"
        "Contexte :\n{context}\n\nSujet : {question}\n\nResume :"
    ),
    "quiz": PromptTemplate.from_template(
        "Tu es enseignant universitaire. Genere exactement 3 QCM en francais.\n"
        "Pour chaque question : enonce, 4 options (A/B/C/D), bonne reponse + justification.\n\n"
        "Contexte :\n{context}\n\nTheme : {question}\n\nQuiz :"
    ),
    "explication": PromptTemplate.from_template(
        "Tu es tuteur pedagogique patient. Explique en 3 niveaux :\n"
        "1) definition simple  2) analogie du quotidien  3) details techniques.\n"
        "Utilise uniquement le contexte. Reponds en francais.\n\n"
        "{history}\n\n"
        "Contexte :\n{context}\n\nConcept : {question}\n\nExplication :"
    ),
}

MODE_LABELS = {
    "Question libre"       : "question",
    "Resume de chapitre"   : "resume",
    "Generer un quiz"      : "quiz",
    "Expliquer un concept" : "explication",
}

# ── Reponse ──────────────────────────────────────────────────────

def normalize_history(history):
    normalized = []
    for item in history or []:
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append({"role": "user", "content": str(item[0])})
            normalized.append({"role": "assistant", "content": str(item[1])})
    return normalized


def answer(question, mode_label, api_key, history):
    global vectorstore_global, conversation_memory

    history = normalize_history(history)

    if not api_key or not api_key.strip():
        history.append({"role": "assistant", "content": "Veuillez entrer votre cle API Groq."})
        return history, ""
    if vectorstore_global is None:
        history.append({"role": "assistant", "content": "Aucun document indexe. Chargez vos PDFs d'abord."})
        return history, ""
    if not question.strip():
        return history, ""

    try:
        mode     = MODE_LABELS.get(mode_label, "question")
        prompt   = PROMPTS[mode]

        retriever = vectorstore_global.as_retriever(search_kwargs={"k": TOP_K})
        raw_docs  = retriever.invoke(question)
        docs      = rerank_docs(raw_docs, question)
        context   = format_context(docs)
        sources   = get_sources(docs)

        history_ctx = ""
        if mode in ("question", "explication"):
            history_ctx = build_history_context(conversation_memory)

        llm          = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=api_key.strip())
        final_prompt = prompt.format(context=context, question=question, history=history_ctx)
        response     = llm.invoke(final_prompt).content.strip()

        sources_str   = "  |  ".join(sources)
        full_response = f"{response}\n\nSources : {sources_str}"

        conversation_memory.append((question, response))
        if len(conversation_memory) > 20:
            conversation_memory[:] = conversation_memory[-20:]

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_response})
        return history, ""

    except Exception as e:
        history.append({"role": "assistant", "content": f"Erreur : {str(e)}"})
        return history, ""

# ── Upload handler ───────────────────────────────────────────────

def upload_pdfs(pdf_files, progress=gr.Progress()):
    global vectorstore_global, conversation_memory
    if not pdf_files:
        return "Aucun fichier selectionne.", gr.update(interactive=False)
    try:
        progress(0.1, desc="Lecture des PDFs...")
        vectorstore_global, n_pages, n_chunks = load_and_index_pdfs(pdf_files)
        conversation_memory = []
        progress(1.0, desc="Indexation terminee !")
        names = [os.path.basename(f.name) for f in pdf_files]
        msg = (
            f"**{len(pdf_files)} fichier(s) indexes !**\n\n"
            f"Fichiers : {', '.join(names)}\n"
            f"Pages chargees : {n_pages} | Chunks : {n_chunks}\n"
            f"Re-ranking : TOP {TOP_K} -> garde {RERANK_KEEP} | "
            f"Memoire : {MAX_HISTORY} tours\n\nPosez vos questions !"
        )
        return msg, gr.update(interactive=True)
    except Exception as e:
        return f"Erreur : {str(e)}", gr.update(interactive=False)

def clear_all():
    global vectorstore_global, conversation_memory
    vectorstore_global  = None
    conversation_memory = []
    return [], "Session reinitialisee."

# ── Interface Gradio ─────────────────────────────────────────────

CSS = """
footer { display: none !important; }
.title { text-align:center; padding:16px 0 8px; }
.title h1 { color:#1a56db; font-size:1.9em; }
.title p  { color:#6b7280; }
"""

with gr.Blocks(title="Assistant Pedagogique IA") as demo:

    gr.HTML("""
    <div class="title">
        <h1>Assistant Pedagogique Intelligent</h1>
       
    </div>""")

    with gr.Row():

        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Configuration")
            api_key_input = gr.Textbox(
                label="Cle API Groq",
                placeholder="gsk_xxxxxxxxxxxxxxxxxxxx",
                type="password",
                info="Gratuit sur console.groq.com",
                value=os.getenv("GROQ_API_KEY", "")
            )
            gr.Markdown("---\n### Vos documents")
            pdf_upload    = gr.File(label="Deposer vos PDFs", file_types=[".pdf"], file_count="multiple")
            upload_btn    = gr.Button("Indexer les documents", variant="primary")
            upload_status = gr.Markdown("Chargez vos PDFs puis cliquez sur Indexer.")

            gr.Markdown("---\n### Mode")
            mode_selector = gr.Radio(
                choices=list(MODE_LABELS.keys()),
                value="Question libre",
                label="",
            )
            gr.Markdown("---")
            clear_btn = gr.Button("Reinitialiser la session", variant="secondary")

          

        with gr.Column(scale=2):
            gr.Markdown("### Conversation")
            chatbot = gr.Chatbot(label="Assistant", height=460)

            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Posez votre question...",
                    label="Votre question",
                    lines=2, scale=5, interactive=False,
                )
                send_btn = gr.Button("Envoyer", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    ["Qu'est-ce qu'un LLM ?",                                  "Question libre"],
                    ["Resume le contenu principal",                             "Resume de chapitre"],
                    ["Genere un quiz sur le prompt engineering",                "Generer un quiz"],
                    ["Explique ce qu'est FAISS",                               "Expliquer un concept"],
                    ["Difference entre zero-shot et few-shot ?",               "Question libre"],
                    ["Explique le chunking et l'overlap",                      "Expliquer un concept"],
                ],
                inputs=[question_input, mode_selector],
                label="Exemples de questions",
            )

    history_state = gr.State([])

    upload_btn.click(
        fn=upload_pdfs, inputs=[pdf_upload],
        outputs=[upload_status, question_input],
    )
    send_btn.click(
        fn=answer,
        inputs=[question_input, mode_selector, api_key_input, history_state],
        outputs=[chatbot, question_input],
    ).then(lambda h: h, inputs=[chatbot], outputs=[history_state])

    question_input.submit(
        fn=answer,
        inputs=[question_input, mode_selector, api_key_input, history_state],
        outputs=[chatbot, question_input],
    ).then(lambda h: h, inputs=[chatbot], outputs=[history_state])

    clear_btn.click(
        fn=clear_all, outputs=[chatbot, upload_status],
    ).then(lambda: [], outputs=[history_state])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
