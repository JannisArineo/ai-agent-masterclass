"""
=============================================================================
LEKTION 2: MEMORY — Wie Agents sich erinnern (OpenAI Version)
=============================================================================

Wichtigster OpenAI Unterschied beim Memory:
- Messages-Format ist identisch zu Anthropic (role/content)
- System Prompt: OpenAI nutzt role="system" als erste Message
  Anthropic hatte einen separaten "system" Parameter
=============================================================================
"""

import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI

client = OpenAI()


class ShortTermMemory:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages = []

    def add(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[:2] + self.messages[-(self.max_messages-2):]

    def get_for_api(self) -> list:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def clear(self):
        self.messages = []


class LongTermMemory:
    def __init__(self, storage_path: str = "memory.json"):
        self.path = Path(storage_path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"facts": {}, "episodes": [], "skills": []}

    def _save(self):
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def store_fact(self, key: str, value: str):
        self.data["facts"][key] = {"value": value, "stored_at": datetime.now().isoformat()}
        self._save()
        print(f"[Memory] Fakt: {key} = {value}")

    def get_fact(self, key: str) -> str | None:
        fact = self.data["facts"].get(key)
        return fact["value"] if fact else None

    def store_episode(self, summary: str, tags: list[str] = None):
        self.data["episodes"].append({
            "summary": summary,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat()
        })
        self.data["episodes"] = self.data["episodes"][-100:]
        self._save()

    def get_context_summary(self) -> str:
        parts = []
        if self.data["facts"]:
            facts_str = "\n".join(f"- {k}: {v['value']}" for k, v in self.data["facts"].items())
            parts.append(f"Bekannte Fakten über den User:\n{facts_str}")
        if self.data["episodes"]:
            recent = self.data["episodes"][-3:]
            episodes_str = "\n".join(f"- {e['summary']}" for e in recent)
            parts.append(f"Letzte Gespräche:\n{episodes_str}")
        return "\n\n".join(parts) if parts else "Keine gespeicherte Memory vorhanden."


class MemoryAgent:
    def __init__(self):
        self.short_term = ShortTermMemory(max_messages=10)
        self.long_term = LongTermMemory("agent_memory.json")

    def _build_messages(self, memory_context: str) -> list:
        """
        OpenAI: System Prompt als erste Message mit role="system"
        Anthropic: separater "system" Parameter im API Call
        """
        system_message = {
            "role": "system",
            "content": f"""Du bist ein hilfreicher AI-Assistent mit Memory.

DEINE ERINNERUNGEN:
{memory_context}

Nutze diese Informationen für personalisierte Antworten."""
        }
        return [system_message] + self.short_term.get_for_api()

    def _extract_facts(self, conversation: str) -> dict:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Du extrahierst Fakten aus Gesprächen als JSON."
                },
                {
                    "role": "user",
                    "content": f"""Extrahiere Fakten über den User aus diesem Gespräch.
Antworte NUR mit JSON: {{"key": "value"}}. Keine Fakten → {{}}

Gespräch:
{conversation}"""
                }
            ]
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {}

    def chat(self, user_message: str) -> str:
        memory_context = self.long_term.get_context_summary()
        self.short_term.add("user", user_message)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self._build_messages(memory_context)
        )

        reply = response.choices[0].message.content
        self.short_term.add("assistant", reply)

        # Fakten aus Gespräch extrahieren + speichern
        new_facts = self._extract_facts(f"User: {user_message}\nAssistent: {reply}")
        for key, value in new_facts.items():
            self.long_term.store_fact(key, value)

        return reply

    def new_session(self, save_episode: bool = True):
        if save_episode and self.short_term.messages:
            convo = "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in self.short_term.messages
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Fasse dieses Gespräch in 1-2 Sätzen zusammen:\n{convo}"}]
            )
            summary = response.choices[0].message.content
            self.long_term.store_episode(summary)
            print(f"[Memory] Session gespeichert: {summary[:80]}...")
        self.short_term.clear()


if __name__ == "__main__":
    agent = MemoryAgent()

    print("=== Session 1 ===")
    print(agent.chat("Hey! Ich bin Jannis, AI Engineer aus München."))
    print(agent.chat("Ich lerne gerade alles über AI Agents."))

    agent.new_session(save_episode=True)

    print("\n=== Session 2 — kennt er mich noch? ===")
    print(agent.chat("Wer bin ich und was mache ich?"))
