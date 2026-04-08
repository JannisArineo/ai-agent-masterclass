"""
=============================================================================
LEKTION 1: DER REACT AGENT — Das Fundament von allem
=============================================================================

ReAct = Reason + Act
Das ist das wichtigste Pattern das du kennen musst.

Die Idee: Anstatt direkt zu antworten, denkt der Agent in Schritten:
    1. THOUGHT  — "Was muss ich tun? Was weiß ich schon?"
    2. ACTION   — "Ich rufe Tool X mit Argument Y auf"
    3. OBSERVATION — "Das Tool hat Z zurückgegeben"
    4. ... wiederholen bis ...
    5. ANSWER   — "Jetzt hab ich genug Info, hier ist die Antwort"

Warum ist das so mächtig?
→ Der Agent kann komplexe Probleme in Schritte zerlegen
→ Er kann Fehler erkennen und korrigieren
→ Er ist nicht auf sein Training-Wissen beschränkt — er kann Tools nutzen
=============================================================================
"""

import json
import anthropic

# Anthropic Client — das Tor zur Claude API
client = anthropic.Anthropic()  # liest ANTHROPIC_API_KEY aus Environment

# =============================================================================
# TOOLS DEFINIEREN
# Tools sind Funktionen die der Agent aufrufen darf.
# Claude bekommt die "Beschreibung" und entscheidet SELBST wann er welches nutzt.
# =============================================================================

TOOLS = [
    {
        "name": "calculator",
        "description": "Führt mathematische Berechnungen durch. Nutze das für alle Rechnungen.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematischer Ausdruck, z.B. '2 + 2' oder '100 * 0.19'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Gibt das aktuelle Wetter für eine Stadt zurück.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name der Stadt"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "search_knowledge",
        "description": "Durchsucht eine interne Wissensdatenbank nach Fakten.",
        "input_schema": {
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
]

# =============================================================================
# TOOL EXECUTOR — die echte Logik hinter den Tools
# In Production würden hier echte APIs angesprochen (OpenWeatherMap, etc.)
# Hier simulieren wir sie für's Lernen.
# =============================================================================

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Führt ein Tool aus und gibt das Ergebnis als String zurück.
    Der Agent sieht nur diesen String — nicht den Code dahinter.
    """

    if tool_name == "calculator":
        try:
            # eval() ist normalerweise gefährlich — hier nur für Demo!
            # In Production: math-Parser-Library nutzen
            result = eval(tool_input["expression"])
            return f"Ergebnis: {result}"
        except Exception as e:
            return f"Fehler bei Berechnung: {e}"

    elif tool_name == "get_weather":
        # Simuliertes Wetter (in echt: API-Call)
        weather_db = {
            "berlin": "18°C, bewölkt, leichter Wind",
            "münchen": "22°C, sonnig",
            "hamburg": "14°C, regnerisch",
        }
        city = tool_input["city"].lower()
        return weather_db.get(city, f"Keine Daten für {tool_input['city']}")

    elif tool_name == "search_knowledge":
        # Simulierte Wissensdatenbank
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


# =============================================================================
# DER AGENT — die Hauptschleife
# Das ist der "Agentic Loop" — läuft so lange bis der Agent fertig ist.
# =============================================================================

def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    Führt den ReAct-Agenten aus.

    Der Loop:
    1. Schicke Nachricht an Claude
    2. Claude antwortet entweder mit:
       a) stop_reason="tool_use" → Agent will ein Tool aufrufen
       b) stop_reason="end_turn" → Agent ist fertig, hat eine Antwort
    3. Bei tool_use: Tool ausführen, Ergebnis zurückschicken → zurück zu 1
    4. Bei end_turn: Fertig!
    """

    # Conversation History — das "Gedächtnis" für diesen Durchlauf
    messages = [{"role": "user", "content": user_message}]

    if verbose:
        print(f"\n{'='*60}")
        print(f"USER: {user_message}")
        print(f"{'='*60}")

    # Agentic Loop — maximal 10 Runden (Schutz vor Endlosschleifen)
    for step in range(10):

        # Claude aufrufen
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Haiku = schnell + günstig für Lernen
            max_tokens=1024,
            tools=TOOLS,           # Tools übergeben — Claude entscheidet ob/welche
            messages=messages
        )

        if verbose:
            print(f"\n[Schritt {step + 1}] stop_reason={response.stop_reason}")

        # --- FALL A: Claude will ein Tool aufrufen ---
        if response.stop_reason == "tool_use":

            # Antwort zur History hinzufügen (wichtig für Kontext!)
            messages.append({"role": "assistant", "content": response.content})

            # Alle Tool-Calls aus der Antwort verarbeiten
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if verbose:
                        print(f"  → Tool: {block.name}({block.input})")

                    # Tool ausführen
                    result = execute_tool(block.name, block.input)

                    if verbose:
                        print(f"  ← Ergebnis: {result}")

                    # Ergebnis sammeln
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,  # ID muss matchen!
                        "content": result
                    })

            # Tool-Ergebnisse zurück an Claude schicken
            messages.append({"role": "user", "content": tool_results})

        # --- FALL B: Claude ist fertig ---
        elif response.stop_reason == "end_turn":
            # Finale Antwort aus dem Text-Block extrahieren
            final_answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_answer += block.text

            if verbose:
                print(f"\nANTWORT: {final_answer}")

            return final_answer

    return "Agent hat maximale Schritte erreicht ohne Antwort."


# =============================================================================
# DEMO — probier's aus
# =============================================================================

if __name__ == "__main__":
    print("\n🤖 ReAct Agent Demo\n")

    # Einfache Rechnung — Agent nutzt calculator
    run_agent("Was ist 347 * 19 + 500?")

    # Mehrere Tools nötig — Agent plant selbst
    run_agent("Wie ist das Wetter in Berlin und was ist 15% von 200 Euro?")

    # Wissen + Berechnung kombiniert
    run_agent("Erkläre mir kurz was ReAct ist und rechne 2^10 aus.")
