# Guinea Intelligence Agent

Agent IA conversationnel en Python, avec mémoire persistante et outils (recherche web, calculatrice, score de crédit simplifié), construit sur l'API Groq (gratuite, sans carte bancaire).

## Fonctionnalités

- 💬 Conversation avec mémoire de session
- 💾 Mémoire persistante (SQLite) — l'agent se souvient entre les sessions
- 🔍 Recherche web (DuckDuckGo, gratuit)
- 🧮 Calculatrice
- 🕒 Date et heure
- 📊 Score de crédit simplifié (démo, lié au projet CreditScore AI Guinée)

## Installation

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Crée un fichier `.env` à la racine avec :