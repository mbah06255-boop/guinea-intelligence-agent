import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOLS_SCHEMA, TOOLS_MAP, log_message, set_current_user

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"
SYSTEM_PROMPT = """Tu es le premier cerveau de l'agent IA de Mouctar, un étudiant ingénieur guinéen basé à Casablanca qui développe des projets fintech pour la Guinée (dont CreditScore AI).

Règles STRICTES sur les outils (à respecter absolument, sans exception) :
- Dès que l'utilisateur te demande de retenir, mémoriser, ou se souvenir d'une information sur le long terme → tu DOIS appeler l'outil save_note. Ne réponds jamais "compris" sans avoir appelé save_note.
- Dès que l'utilisateur te demande ce que tu as retenu / mémorisé / sauvegardé → tu DOIS appeler l'outil get_notes avant de répondre, ne réponds jamais depuis ta mémoire de conversation.
- Dès qu'il s'agit d'évaluer un profil emprunteur (revenu, dettes, épargne, ancienneté) → tu DOIS utiliser calculate_credit_score, JAMAIS calculator pour ce cas précis.
- Utilise calculator uniquement pour des calculs mathématiques simples et génériques, pas pour un score de crédit.
- Utilise web_search uniquement pour des informations actuelles ou incertaines. Jamais plus de 2 recherches pour la même question.
- Utilise get_datetime si on te demande la date ou l'heure actuelle."""
MAX_TOOL_ITERATIONS = 3
MAX_TURNS = 12  # nombre d'échanges complets conservés dans l'historique


class Agent:
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.system_message = {"role": "system", "content": SYSTEM_PROMPT}
        self.turns = []

    def _flatten(self):
        messages = [self.system_message]
        for turn in self.turns:
            messages.extend(turn)
        return messages

    def ask(self, user_message: str) -> str:
        set_current_user(self.user_id)
        turn = [{"role": "user", "content": user_message}]
        # ... reste identique

        for _ in range(MAX_TOOL_ITERATIONS):
            response = client.chat.completions.create(
                model=MODEL,
                messages=self._flatten() + turn,
                tools=TOOLS_SCHEMA
            )
            message = response.choices[0].message

            if message.tool_calls:
                turn.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                for tool_call in message.tool_calls:
                    func = TOOLS_MAP.get(tool_call.function.name)
                    args = json.loads(tool_call.function.arguments or "{}")
                    try:
                        result = func(**args) if func else f"Outil inconnu : {tool_call.function.name}"
                    except Exception as e:
                        result = f"Erreur pendant l'exécution de l'outil : {e}"

                    turn.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                continue

            reply = message.content or "(Réponse vide reçue du modèle)"
            turn.append({"role": "assistant", "content": reply})
            log_message("user", user_message)
            log_message("assistant", reply)
            self.turns.append(turn)
            self._trim_history()
            return reply

        turn.append({"role": "assistant", "content": "Désolé, je n'ai pas réussi à conclure après plusieurs tentatives d'outils."})
        log_message("user", user_message)
        log_message("assistant", turn[-1]["content"])
        self.turns.append(turn)
        self._trim_history()
        return turn[-1]["content"]

    def _trim_history(self):
        if len(self.turns) > MAX_TURNS:
            self.turns = self.turns[-MAX_TURNS:]