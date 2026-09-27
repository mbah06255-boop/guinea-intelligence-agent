import os
import sqlite3
from datetime import datetime
from ddgs import DDGS
import joblib
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "credit_model.pkl")

# ---- Utilisateur courant (défini par l'Agent avant chaque appel d'outil) ----
_current_user_id = "default"


def set_current_user(user_id: str) -> None:
    global _current_user_id
    _current_user_id = user_id or "default"


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


_init_db()

try:
    _credit_model = joblib.load(MODEL_PATH)
    print(f"✅ Modèle XGBoost chargé depuis {MODEL_PATH}")
except Exception:
    _credit_model = None
    print("⚠️ Aucun modèle XGBoost trouvé — utilisation de la formule simplifiée en secours.")


# ---- Recherche web (avec gestion d'erreurs) ----
def web_search(query: str) -> str:
    print(f"   🔍 [Recherche web : {query}]")
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Erreur pendant la recherche web (réessaie plus tard) : {e}"

    if not results:
        return "Aucun résultat trouvé."
    return "\n".join(f"- {r['title']}: {r['body']} ({r['href']})" for r in results)


# ---- Calculatrice ----
def calculator(expression: str) -> str:
    print(f"   🧮 [Calcul : {expression}]")
    allowed = "0123456789+-*/(). "
    if not all(c in allowed for c in expression):
        return "Expression invalide : caractères non autorisés."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Erreur de calcul : {e}"


# ---- Date/heure ----
def get_datetime() -> str:
    print("   🕒 [Date/heure demandée]")
    return datetime.now().strftime("%A %d %B %Y, %H:%M:%S")


# ---- Mémoire persistante (SQLite, isolée par utilisateur) ----
def save_note(content: str) -> str:
    print(f"   💾 [Sauvegarde note pour {_current_user_id} : {content}]")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO notes (user_id, content, created_at) VALUES (?, ?, ?)",
        (_current_user_id, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return "Note sauvegardée avec succès."


def get_notes() -> str:
    print(f"   📖 [Lecture des notes pour {_current_user_id}]")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT content, created_at FROM notes WHERE user_id = ? ORDER BY id DESC LIMIT 10",
        (_current_user_id,)
    ).fetchall()
    conn.close()

    if not rows:
        return "Aucune note enregistrée pour l'instant."
    return "\n".join(f"- [{created}] {content}" for content, created in rows)


# ---- Score de crédit (vrai modèle XGBoost, avec secours formule simplifiée) ----
def calculate_credit_score(
    revenu_mensuel_gnf: float,
    dettes_mensuelles_gnf: float,
    epargne_mensuelle_gnf: float,
    anciennete_activite_mois: int
) -> str:
    print("   📊 [Calcul du score de crédit]")

    if revenu_mensuel_gnf <= 0:
        return "Revenu mensuel invalide (doit être supérieur à 0)."

    if _credit_model is not None:
        X = pd.DataFrame([{
            "revenu_mensuel_gnf": revenu_mensuel_gnf,
            "dettes_mensuelles_gnf": dettes_mensuelles_gnf,
            "epargne_mensuelle_gnf": epargne_mensuelle_gnf,
            "anciennete_activite_mois": anciennete_activite_mois
        }])
        default_proba = _credit_model.predict_proba(X)[0][1]
        score = round((1 - default_proba) * 100)
        source = "modèle XGBoost entraîné"
    else:
        ratio_endettement = dettes_mensuelles_gnf / revenu_mensuel_gnf
        ratio_epargne = epargne_mensuelle_gnf / revenu_mensuel_gnf
        score = 100
        score -= min(ratio_endettement * 100, 50)
        score += min(ratio_epargne * 50, 20)
        score += min(anciennete_activite_mois / 2, 15)
        score = max(0, min(100, round(score)))
        source = "formule simplifiée (modèle non trouvé)"

    if score >= 70:
        niveau = "Faible risque"
    elif score >= 40:
        niveau = "Risque modéré"
    else:
        niveau = "Risque élevé"

    return (
        f"Score de crédit : {score}/100 — {niveau} (via {source})\n"
        f"Profil : revenu {revenu_mensuel_gnf:,.0f} GNF, dettes {dettes_mensuelles_gnf:,.0f} GNF, "
        f"épargne {epargne_mensuelle_gnf:,.0f} GNF, ancienneté {anciennete_activite_mois} mois."
    )


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Recherche des informations actuelles sur le web. À utiliser pour l'actualité, des faits récents, ou des données que tu ne connais pas avec certitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Les mots-clés à rechercher"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Effectue un calcul mathématique précis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expression mathématique, ex: '12 * (3 + 4)'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Retourne la date et l'heure actuelles.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Sauvegarde une information importante en mémoire persistante, pour s'en souvenir même après avoir fermé le programme.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Le contenu à mémoriser"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_notes",
            "description": "Récupère les notes précédemment sauvegardées en mémoire persistante.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_credit_score",
            "description": "Calcule un score de crédit pour un profil de microfinance en Guinée, à partir du revenu, des dettes, de l'épargne mensuelle et de l'ancienneté d'activité, via un modèle XGBoost entraîné.",
            "parameters": {
                "type": "object",
                "properties": {
                    "revenu_mensuel_gnf": {"type": "number", "description": "Revenu mensuel en francs guinéens"},
                    "dettes_mensuelles_gnf": {"type": "number", "description": "Dettes mensuelles en francs guinéens"},
                    "epargne_mensuelle_gnf": {"type": "number", "description": "Épargne mensuelle en francs guinéens"},
                    "anciennete_activite_mois": {"type": "integer", "description": "Ancienneté de l'activité/business en mois"}
                },
                "required": ["revenu_mensuel_gnf", "dettes_mensuelles_gnf", "epargne_mensuelle_gnf", "anciennete_activite_mois"]
            }
        }
    }
]

TOOLS_MAP = {
    "web_search": web_search,
    "calculator": calculator,
    "get_datetime": get_datetime,
    "save_note": save_note,
    "get_notes": get_notes,
    "calculate_credit_score": calculate_credit_score,
}


def log_message(role: str, content: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO conversations (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (_current_user_id, role, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()