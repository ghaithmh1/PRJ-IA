# 🎓 Assistant Pédagogique Intelligent

Mini-projet IA Générative — Module IA Générative  
**Stack :** Gradio · LangChain · FAISS · Groq

---

## Description

Application web intelligente permettant aux étudiants de :
- Charger leurs cours en PDF
- Poser des questions sur le contenu
- Obtenir des résumés de chapitres
- Générer des QCM d'entraînement
- Faire expliquer des concepts en détail

---

## Architecture

```
Étudiant (Gradio UI)
      ↓
 Question + Mode
      ↓
 Retriever FAISS  ←──  PDF indexés (LangChain + HuggingFace Embeddings)
      ↓
 Top-K chunks pertinents
      ↓
 PromptTemplate (adapté au mode)
      ↓
 ChatGroq (llama-3.3-70b-versatile)
      ↓
 Réponse affichée dans le chat
```

---

## Installation

```bash
# 1. Créer un environnement virtuel
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Créer le fichier .env
echo "GROQ_API_KEY=votre_cle_ici" > .env

# 4. Lancer l'application
python app.py
```

Ouvrir ensuite : http://localhost:7860

---

## Obtenir une clé Groq (gratuit)

1. Aller sur https://console.groq.com
2. Créer un compte
3. Cliquer sur **Create API Key**
4. Copier la clé dans le fichier `.env` ou la coller dans l'interface

---

## Modes disponibles

| Mode | Description |
|------|-------------|
| ❓ Question libre | Pose une question sur tes cours |
| 📄 Résumé de chapitre | Résumé structuré d'un sujet |
| 🧪 Générer un quiz | 3 QCM avec corrections |
| 💡 Expliquer un concept | Explication progressive + analogie |

---

## Paramètres RAG

| Paramètre | Valeur | Explication |
|-----------|--------|-------------|
| `CHUNK_SIZE` | 800 | Taille de chaque extrait de texte |
| `CHUNK_OVERLAP` | 150 | Chevauchement entre extraits |
| `TOP_K` | 4 | Nombre d'extraits récupérés par question |
| Embedding model | `all-MiniLM-L6-v2` | Modèle léger et efficace |
| LLM | `llama-3.3-70b-versatile` | Modèle Groq puissant |

---

## Structure du projet

```
assistant_pedagogique/
├── app.py              ← Application principale
├── requirements.txt    ← Dépendances Python
├── .env                ← Clé API (à créer)
└── README.md           ← Ce fichier
```

---

## Concepts mis en œuvre (lien avec les TPs)

- **TP1** : Prompt Engineering — 4 PromptTemplates différents selon le mode
- **TP2** : RAG complet — LangChain + FAISS + Groq + chunking
- **TP3** : Interface Gradio — upload de fichiers, chatbot, exemples interactifs
