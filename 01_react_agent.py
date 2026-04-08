"""
=============================================================================
LEKTION 1: DER REACT AGENT — Das Fundament von allem
=============================================================================

ReAct = Reason + Act

OPENAI vs ANTHROPIC — die wichtigsten Unterschiede:
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ Feature             │ OpenAI                   │ Anthropic                │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Tool-Schema         │ {"type":"function",...}  │ {"name":..,"input_schema"│
│ Tool-Aufruf Signal  │ finish_reason="tool_calls│ stop_reason="tool_use"   │
│ Tool-Ergebnis Role  │ role="tool"              │ role="user" (als Liste)  │
│ Tool-Args Format    │ json.loads(tc.function   │ block.input (dict)       │
│                     │ .arguments)              │                          │
│ Response Text       │ choices[0].message       │ response.content[0].text │
│ Token Tracking      │ prompt_tokens            │ input_tokens             │
└─────────────────────┴──────────────────────────┴──────────────────────────┘
=============================================================================
"""

import json
from openai import OpenAI

# OpenAI Client — liest OPENAI_API_KEY aus Environment
client = OpenAI()

# =============================================================================
# TOOLS DEFINIEREN
# OpenAI Format: Tool ist in ein "function"-Objekt eingewickelt
# Anthropic hatte direkt "input_schema" — hier heißt es "parameters"
# =============================================================================

TOOLS = [
    {
        "type": "function",                          # OpenAI braucht diesen Wrapper
        "function": {
            "name": "calculator",
            "description": "Führt mathematische Berechnungen durch. Nutze das für alle Rechnungen.",
            "parameters": {                          # Anthropic: "input_schema"
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematischer Ausdruck, z.B. '2 + 2' oder '100 * 0.19'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gibt das aktuelle Wetter für eine Stadt zurück.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name der Stadt"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Durchsucht eine interne Wissensdatenbank nach Fakten.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Suchanfrage"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "calculator":
        try:
            result = eval(tool_input["expression"])
            return f"Ergebnis: {result}"
        except Exception as e:
            return f"Fehler bei Berechnung: {e}"

    elif tool_name == "get_weather":
        weather_db = {
            "berlin": "18°C, bewölkt, leichter Wind",
            "münchen": "22°C, sonnig",
            "hamburg": "14°C, regnerisch",
        }
        return weather_db.get(tool_input["city"].lower(), f"Keine Daten für {tool_input['city']}")

    elif tool_name == "search_knowledge":
        knowledge = {
            "python": "Python ist eine interpretierte Hochsprache, erstellt 1991 von Guido van Rossum.",
            "claude": "Claude ist ein KI-Assistent von Anthropic, erschienen 2023.",
            "react": "ReAct ist ein Agent-Pattern: Reason + Act, veröffentlicht 2022 von Google Research.",
        }
        query = tool_input["query"].lower()
        for key, value in knowledge.items():
            if key in query:
                return value
        return "Keine Informationen gefunden."

    return f"Unbekanntes Tool: {tool_name}"


def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    ReAct Loop mit OpenAI.

    OpenAI Flow:
    1. API Call
    2. finish_reason == "tool_calls"  → Tool ausführen
       finish_reason == "stop"        → Fertig (Anthropic: "end_turn")
    3. Tool-Ergebnis als role="tool" Message zurückschicken
       (Anthropic: als role="user" mit type="tool_result")
    """
    messages = [{"role": "user", "content": user_message}]

    if verbose:
        print(f"\n{'='*60}\nUSER: {user_message}\n{'='*60}")

    for step in range(10):
        response = client.chat.completions.create(
            model="gpt-4o-mini",        # günstig + schnell, wie claude-haiku
            tools=TOOLS,
            messages=messages
        )

        message = response.choices[0].message    # OpenAI: choices[0].message
        finish_reason = response.choices[0].finish_reason

        if verbose:
            print(f"\n[Schritt {step + 1}] finish_reason={finish_reason}")

        # --- FALL A: Tool aufrufen ---
        if finish_reason == "tool_calls":         # Anthropic: "tool_use"
            messages.append(message)              # Assistenten-Message zur History

            for tc in message.tool_calls:         # Anthropic: response.content blocks
                tool_name = tc.function.name
                tool_input = json.loads(tc.function.arguments)  # Anthropic: block.input (dict)

                if verbose:
                    print(f"  → Tool: {tool_name}({tool_input})")

                result = execute_tool(tool_name, tool_input)

                if verbose:
                    print(f"  ← Ergebnis: {result}")

                # Tool-Ergebnis zurückschicken
                # OpenAI: eigene "tool" role
                # Anthropic: role="user" mit type="tool_result" in einer Liste
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,        # muss zur tool_call ID passen
                    "content": result
                })

        # --- FALL B: Fertig ---
        elif finish_reason == "stop":             # Anthropic: "end_turn"
            final = message.content or ""
            if verbose:
                print(f"\nANTWORT: {final}")
            return final

    return "Maximale Schritte erreicht."


if __name__ == "__main__":
    print("\n🤖 ReAct Agent Demo (OpenAI)\n")
    run_agent("Was ist 347 * 19 + 500?")
    run_agent("Wie ist das Wetter in Berlin und was ist 15% von 200 Euro?")
