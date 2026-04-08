"""
=============================================================================
LEKTION 2: MEMORY — Wie Agents sich erinnern
=============================================================================

Das ist der Unterschied zwischen einem dummen Chatbot und einem echten Agent.

Es gibt 4 Typen von Memory:

1. IN-CONTEXT MEMORY (kurzfristig)
   → Die Conversation History im aktuellen Request
   → Limitiert durch das Context Window (200k Tokens bei Claude)
   → Geht verloren wenn der Prozess endet

2. EXTERNAL MEMORY (langfristig)
   → Gespeichert in einer Datenbank / Datei
   → Überlebt Neustarts
   → Kann mit Semantic Search durchsucht werden (Vector DB)

3. IN-WEIGHTS MEMORY
   → Das was Claude im Training gelernt hat
   → Kann nicht geändert werden (ohne Fine-Tuning)

4. IN-CACHE MEMORY
   → Prompt Caching für wiederkehrende System Prompts
   → Spart Kosten und Speed

In dieser Lektion: Typ 1 + 2 — die die du wirklich baust.
=============================================================================
"""

import json
import os
from datetime import datetime
from pathlib import Path

import anthropic

client = anthropic.Anthropic()

# =============================================================================
# KURZZEIT-MEMORY: Conversation History
# Einfachste Form — einfach alle Messages in einer Liste speichern
# =============================================================================

class ShortTermMemory:
    """
    Verwaltet die aktuelle Conversation History.
    Das ist das "Working Memory" des Agents — was er gerade im Kopf hat.
    """

    def __init__(self, max_messages: int = 20):
        # max_messages verhindert dass das Context Window überläuft
        self.max_messages = max_messages
        self.messages = []

    def add(self, role: str, content: str):
        """Fügt eine Nachricht zur History hinzu."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Älteste Nachrichten löschen wenn zu viele (Sliding Window)
        if len(self.messages) > self.max_messages:
            # Erste 2 behalten (oft System-Kontext), Rest kürzen
            self.messages = self.messages[:2] + self.messages[-(self.max_messages-2):]

    def get_for_api(self) -> list:
        """Gibt die Messages im Format zurück das die API erwartet."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def clear(self):
        self.messages = []


# =============================================================================
# LANGZEIT-MEMORY: Persistent Storage
# Hier nutzen wir einfache JSON-Dateien — in Production: PostgreSQL, Redis, etc.
# =============================================================================

class LongTermMemory:
    """
    Persistente Memory — überlebt Neustarts.

    Struktur:
    - "facts": Fakten über den User (Name, Präferenzen, etc.)
    - "episodes": Vergangene Gespräche (zusammengefasst)
    - "skills": Was der Agent gelernt hat zu tun

    In einer echten App würdest du hier eine Vector DB nutzen (Pinecone, Weaviate)
    für semantische Suche statt einfacher Keyword-Suche.
    """

    def __init__(self, storage_path: str = "memory.json"):
        self.path = Path(storage_path)
        self.data = self._load()

    def _load(self) -> dict:
        """Lädt Memory aus Datei oder erstellt leere Struktur."""
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"facts": {}, "episodes": [], "skills": []}

    def _save(self):
        """Speichert Memory auf Disk."""
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def store_fact(self, key: str, value: str):
        """Speichert einen Fakt über den User."""
        self.data["facts"][key] = {
            "value": value,
            "stored_at": datetime.now().isoformat()
        }
        self._save()
        print(f"[Memory] Fakt gespeichert: {key} = {value}")

    def get_fact(self, key: str) -> str | None:
        """Holt einen gespeicherten Fakt."""
        fact = self.data["facts"].get(key)
        return fact["value"] if fact else None

    def store_episode(self, summary: str, tags: list[str] = None):
        """Speichert eine Zusammenfassung eines Gesprächs."""
        self.data["episodes"].append({
            "summary": summary,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat()
        })
        # Maximal 100 Episoden behalten
        self.data["episodes"] = self.data["episodes"][-100:]
        self._save()

    def search_episodes(self, query: str) -> list[str]:
        """
        Primitive Suche — schaut ob query-Wörter in den Tags/Summaries vorkommen.
        In Production: Vector Embedding + Cosine Similarity für echte semantische Suche.
        """
        query_words = query.lower().split()
        results = []
        for episode in self.data["episodes"]:
            text = (episode["summary"] + " " + " ".join(episode["tags"])).lower()
            if any(word in text for word in query_words):
                results.append(episode["summary"])
        return results[-5:]  # Top 5 zurückgeben

    def get_context_summary(self) -> str:
        """
        Erstellt einen Kontext-String aus der Memory.
        Dieser wird dem Agent als zusätzlicher Kontext gegeben.
        """
        parts = []

        if self.data["facts"]:
            facts_str = "\n".join(f"- {k}: {v['value']}" for k, v in self.data["facts"].items())
            parts.append(f"Bekannte Fakten über den User:\n{facts_str}")

        if self.data["episodes"]:
            recent = self.data["episodes"][-3:]  # Letzte 3 Episoden
            episodes_str = "\n".join(f"- {e['summary']}" for e in recent)
            parts.append(f"Letzte Gespräche:\n{episodes_str}")

        return "\n\n".join(parts) if parts else "Keine gespeicherte Memory vorhanden."


# =============================================================================
# MEMORY-AUGMENTED AGENT
# Ein Agent der beide Memory-Typen kombiniert
# =============================================================================

class MemoryAgent:
    """
    Agent mit vollständigem Memory-System.

    Der Flow:
    1. User schickt Nachricht
    2. Agent lädt relevante Long-Term Memory
    3. Fügt Memory als Kontext zum System Prompt hinzu
    4. Führt Conversation
    5. Extrahiert neue Fakten aus der Conversation
    6. Speichert Episode in Long-Term Memory
    """

    def __init__(self):
        self.short_term = ShortTermMemory(max_messages=10)
        self.long_term = LongTermMemory("agent_memory.json")

    def _extract_facts(self, conversation: str) -> dict:
        """
        Nutzt Claude um Fakten aus einem Gespräch zu extrahieren.
        Das ist 'Memory Consolidation' — wie das menschliche Gehirn im Schlaf.
        """
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""Analysiere dieses Gespräch und extrahiere wichtige Fakten über den User.
Antworte NUR mit JSON im Format {{"key": "value"}}.
Beispiele: {{"name": "Max", "beruf": "Entwickler", "lieblingssprache": "Python"}}
Wenn keine Fakten erkennbar: antworte mit {{}}

Gespräch:
{conversation}"""
            }]
        )
        try:
            return json.loads(response.content[0].text)
        except:
            return {}

    def chat(self, user_message: str) -> str:
        """Hauptmethode — verarbeitet eine User-Nachricht."""

        # Long-Term Memory als Kontext laden
        memory_context = self.long_term.get_context_summary()

        # System Prompt mit Memory anreichern
        system_prompt = f"""Du bist ein hilfreicher AI-Assistent mit Memory.

DEINE ERINNERUNGEN:
{memory_context}

Nutze diese Informationen um personalisierte Antworten zu geben.
Wenn der User neue Informationen über sich teilt, merke sie dir."""

        # Short-Term Memory mit neuer Nachricht
        self.short_term.add("user", user_message)

        # API Call
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=self.short_term.get_for_api()
        )

        assistant_reply = response.content[0].text
        self.short_term.add("assistant", assistant_reply)

        # Im Hintergrund: Fakten aus dem Gespräch extrahieren + speichern
        # (in Production: async, nicht blocking)
        conversation_text = f"User: {user_message}\nAssistent: {assistant_reply}"
        new_facts = self._extract_facts(conversation_text)
        for key, value in new_facts.items():
            self.long_term.store_fact(key, value)

        return assistant_reply

    def new_session(self, save_episode: bool = True):
        """Startet eine neue Session — speichert die alte als Episode."""
        if save_episode and self.short_term.messages:
            # Gespräch zusammenfassen und in Long-Term speichern
            convo = "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in self.short_term.messages
            )
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"Fasse dieses Gespräch in 1-2 Sätzen zusammen:\n{convo}"
                }]
            )
            summary = response.content[0].text
            self.long_term.store_episode(summary)
            print(f"[Memory] Session gespeichert: {summary[:80]}...")

        self.short_term.clear()


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    agent = MemoryAgent()

    print("=== Session 1: Jannis stellt sich vor ===")
    print(agent.chat("Hey! Ich bin Jannis, AI Engineer aus München."))
    print(agent.chat("Ich lerne gerade alles über AI Agents und arbeite viel mit Python."))

    # Session speichern und neue starten
    agent.new_session(save_episode=True)

    print("\n=== Session 2: Neue Session — kennt er mich noch? ===")
    print(agent.chat("Hey, wer bin ich nochmal und was mache ich beruflich?"))
