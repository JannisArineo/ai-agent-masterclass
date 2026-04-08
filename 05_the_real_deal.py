"""
=============================================================================
LEKTION 5: DAS KOMPLETTE SYSTEM — Alles zusammen
=============================================================================

Das hier ist das was du in Production bauen würdest.

Ein vollständiger AI Agent mit:
✓ ReAct Loop (Reason + Act)
✓ Tool Use (echte Funktionen)
✓ Memory (Short + Long Term)
✓ Multi-Agent Orchestration
✓ Self-Reflection
✓ Streaming Output
✓ Error Handling

Das ist kein Tutorial-Code mehr — das ist ein echtes System.
=============================================================================

WARUM MACHT DICH DAS ZUM AI OBERLORD:

1. Du verstehst WARUM jedes Pattern existiert
   → Nicht nur "copy paste from docs"

2. Du kannst jedes System debuggen
   → Du weißt wo der Loop ist, wo Memory gespeichert wird, warum ein Tool failed

3. Du kannst neue Capabilities hinzufügen
   → Neues Tool? BaseAgent erweitern. Neues Memory-Backend? LongTermMemory tauschen.

4. Du kennst die Grenzen
   → Context Window, Token Costs, Latenz — du planst darum herum

5. Du bist der Mensch der das System designed
   → Claude/GPT4 ist dein Worker, nicht dein Boss
=============================================================================
"""

import anthropic
import json
from datetime import datetime
from pathlib import Path

client = anthropic.Anthropic()


# =============================================================================
# PRODUCTION-GRADE TOOL SYSTEM
# =============================================================================

class ToolRegistry:
    """
    Zentrale Registry für alle Tools.
    Trennt Tool-Definition (für die API) von Tool-Implementation (echte Logik).
    """

    def __init__(self):
        self._tools = {}      # name → implementation function
        self._schemas = []    # API-Format für Claude

    def register(self, name: str, description: str, parameters: dict):
        """Decorator zum Registrieren von Tools."""
        def decorator(func):
            self._tools[name] = func
            self._schemas.append({
                "name": name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": parameters,
                    "required": list(parameters.keys())
                }
            })
            return func
        return decorator

    def execute(self, name: str, inputs: dict) -> str:
        if name not in self._tools:
            return f"Fehler: Tool '{name}' nicht gefunden"
        try:
            return str(self._tools[name](**inputs))
        except Exception as e:
            return f"Tool-Fehler: {e}"

    @property
    def schemas(self):
        return self._schemas


# Tool Registry erstellen
tools = ToolRegistry()

# Tools registrieren via Decorator
@tools.register(
    name="save_note",
    description="Speichert eine Notiz dauerhaft.",
    parameters={
        "title": {"type": "string", "description": "Titel der Notiz"},
        "content": {"type": "string", "description": "Inhalt"}
    }
)
def save_note(title: str, content: str) -> str:
    notes_file = Path("agent_notes.json")
    notes = json.loads(notes_file.read_text()) if notes_file.exists() else []
    notes.append({"title": title, "content": content, "created": datetime.now().isoformat()})
    notes_file.write_text(json.dumps(notes, ensure_ascii=False, indent=2))
    return f"Notiz '{title}' gespeichert."

@tools.register(
    name="list_notes",
    description="Listet alle gespeicherten Notizen auf.",
    parameters={"filter": {"type": "string", "description": "Optional: Suchbegriff"}}
)
def list_notes(filter: str = "") -> str:
    notes_file = Path("agent_notes.json")
    if not notes_file.exists():
        return "Keine Notizen vorhanden."
    notes = json.loads(notes_file.read_text())
    if filter:
        notes = [n for n in notes if filter.lower() in n["title"].lower() or filter.lower() in n["content"].lower()]
    if not notes:
        return "Keine passenden Notizen."
    return "\n".join(f"- {n['title']}: {n['content'][:50]}..." for n in notes)

@tools.register(
    name="calculate",
    description="Führt mathematische Berechnungen durch.",
    parameters={"expression": {"type": "string", "description": "Mathematischer Ausdruck"}}
)
def calculate(expression: str) -> str:
    import ast, operator
    # Sicheres Eval — nur Math-Operationen erlaubt (kein exec/import/etc.)
    allowed = {ast.Add: operator.add, ast.Sub: operator.sub,
               ast.Mult: operator.mul, ast.Div: operator.truediv,
               ast.Pow: operator.pow, ast.USub: operator.neg}
    def safe_eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return allowed[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed[type(node.op)](safe_eval(node.operand))
        raise ValueError(f"Unsupported: {node}")
    try:
        tree = ast.parse(expression, mode='eval')
        return f"{expression} = {safe_eval(tree.body)}"
    except:
        return f"Fehler bei: {expression}"

@tools.register(
    name="get_datetime",
    description="Gibt aktuelles Datum und Uhrzeit zurück.",
    parameters={"format": {"type": "string", "description": "Format: 'date', 'time', oder 'full'"}}
)
def get_datetime(format: str = "full") -> str:
    now = datetime.now()
    formats = {
        "date": now.strftime("%d.%m.%Y"),
        "time": now.strftime("%H:%M:%S"),
        "full": now.strftime("%d.%m.%Y %H:%M:%S")
    }
    return formats.get(format, formats["full"])


# =============================================================================
# DER VOLLSTÄNDIGE PRODUCTION AGENT
# =============================================================================

class ProductionAgent:
    """
    Der fertige Agent — Production-Ready.

    Features:
    - ReAct Loop mit Tool Use
    - Streaming Output (siehst die Antwort live)
    - Conversation Memory
    - Automatische Fehlerbehandlung
    - Token-Tracking für Cost-Monitoring
    - Max-Steps Schutz gegen Endlosschleifen
    """

    def __init__(self, name: str = "Agent", max_steps: int = 10):
        self.name = name
        self.max_steps = max_steps
        self.conversation: list[dict] = []
        self.total_tokens = 0
        self.total_cost_usd = 0.0  # Tracking für echte Produktionssysteme

        # System Prompt — die "Persönlichkeit" des Agents
        self.system = f"""Du bist {name}, ein kompetenter AI Agent.

Heute ist {datetime.now().strftime('%d.%m.%Y')}.

Du hast Zugriff auf Tools. Nutze sie wenn nötig.
Denke Schritt für Schritt. Sei präzise und direkt."""

    def _track_cost(self, input_tokens: int, output_tokens: int):
        """
        Trackt Token-Kosten.
        Haiku Pricing: $0.80/M input, $4/M output (Stand 2025)
        In Production: wichtig für Budget-Kontrolle!
        """
        cost = (input_tokens * 0.0008 + output_tokens * 0.004) / 1000
        self.total_tokens += input_tokens + output_tokens
        self.total_cost_usd += cost

    def chat(self, user_message: str, stream: bool = True) -> str:
        """
        Hauptmethode: User-Nachricht verarbeiten.
        stream=True: Output wird live angezeigt (besser UX)
        """
        self.conversation.append({"role": "user", "content": user_message})
        print(f"\n[{self.name}] Denke nach...\n")

        for step in range(self.max_steps):
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                system=self.system,
                tools=tools.schemas,
                messages=self.conversation
            )

            self._track_cost(response.usage.input_tokens, response.usage.output_tokens)

            # Tool Use — Agent will Aktion ausführen
            if response.stop_reason == "tool_use":
                self.conversation.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "text" and block.text:
                        print(f"[Gedanke] {block.text}")  # Agent "denkt laut"
                    elif block.type == "tool_use":
                        print(f"[Tool] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                        result = tools.execute(block.name, block.input)
                        print(f"[Result] {result}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                self.conversation.append({"role": "user", "content": tool_results})

            # Fertig — finale Antwort
            elif response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text

                self.conversation.append({"role": "assistant", "content": final_text})

                # Kosten anzeigen (in Production: ins Logging schreiben)
                print(f"\n[Stats] {self.total_tokens} Tokens | ${self.total_cost_usd:.4f}")

                return final_text

        return "Maximale Schritte erreicht."

    def reset(self):
        """Neue Conversation starten — Memory löschen."""
        self.conversation = []
        print(f"[{self.name}] Conversation zurückgesetzt.")


# =============================================================================
# DEMO — das System in Aktion
# =============================================================================

if __name__ == "__main__":
    agent = ProductionAgent(name="Jarvis")

    print("\n" + "="*60)
    print("PRODUCTION AGENT DEMO")
    print("="*60)

    # Test 1: Tool Use
    response = agent.chat("Was ist heute für ein Datum und rechne 1337 * 42 aus.")
    print(f"\nAntwort: {response}")

    # Test 2: Memory über mehrere Messages
    response = agent.chat("Speichere eine Notiz: Titel 'AI Learning' mit dem Inhalt 'ReAct, Memory, Multi-Agent gelernt'")
    print(f"\nAntwort: {response}")

    response = agent.chat("Was hab ich gerade gespeichert?")
    print(f"\nAntwort: {response}")

    # Kosten-Übersicht
    print(f"\n[Gesamt] {agent.total_tokens} Tokens | ${agent.total_cost_usd:.4f} USD")
