"""
=============================================================================
LEKTION 5: DAS KOMPLETTE SYSTEM — Production-Grade (OpenAI Version)
=============================================================================

OpenAI Tool-Call Flow:
1. API Call mit tools=[...]
2. finish_reason == "tool_calls"  → Tools ausführen
3. Tool-Ergebnis als role="tool" Message zurückschicken
4. Zurück zu 1 bis finish_reason == "stop"

Token Kosten gpt-4o-mini (Stand 2025):
- Input:  $0.15 / 1M Tokens
- Output: $0.60 / 1M Tokens
=============================================================================
"""

import json
import ast
import operator
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._schemas = []

    def register(self, name: str, description: str, parameters: dict):
        def decorator(func):
            self._tools[name] = func
            # OpenAI: Tool in "function"-Objekt einwickeln
            self._schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": list(parameters.keys())
                    }
                }
            })
            return func
        return decorator

    def execute(self, name: str, inputs: dict) -> str:
        if name not in self._tools:
            return f"Tool '{name}' nicht gefunden"
        try:
            return str(self._tools[name](**inputs))
        except Exception as e:
            return f"Fehler: {e}"

    @property
    def schemas(self):
        return self._schemas


tools = ToolRegistry()

@tools.register(
    name="save_note",
    description="Speichert eine Notiz dauerhaft.",
    parameters={
        "title": {"type": "string", "description": "Titel"},
        "content": {"type": "string", "description": "Inhalt"}
    }
)
def save_note(title: str, content: str) -> str:
    path = Path("agent_notes.json")
    notes = json.loads(path.read_text()) if path.exists() else []
    notes.append({"title": title, "content": content, "created": datetime.now().isoformat()})
    path.write_text(json.dumps(notes, ensure_ascii=False, indent=2))
    return f"Notiz '{title}' gespeichert."

@tools.register(
    name="list_notes",
    description="Listet gespeicherte Notizen auf.",
    parameters={"filter": {"type": "string", "description": "Optionaler Suchbegriff"}}
)
def list_notes(filter: str = "") -> str:
    path = Path("agent_notes.json")
    if not path.exists():
        return "Keine Notizen."
    notes = json.loads(path.read_text())
    if filter:
        notes = [n for n in notes if filter.lower() in n["title"].lower() or filter.lower() in n["content"].lower()]
    return "\n".join(f"- {n['title']}: {n['content'][:50]}..." for n in notes) or "Keine Treffer."

@tools.register(
    name="calculate",
    description="Sichere mathematische Berechnungen.",
    parameters={"expression": {"type": "string", "description": "Mathematischer Ausdruck"}}
)
def calculate(expression: str) -> str:
    allowed = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg
    }
    def safe_eval(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp): return allowed[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        if isinstance(node, ast.UnaryOp): return allowed[type(node.op)](safe_eval(node.operand))
        raise ValueError(f"Nicht erlaubt: {node}")
    try:
        return f"{expression} = {safe_eval(ast.parse(expression, mode='eval').body)}"
    except:
        return f"Fehler bei: {expression}"

@tools.register(
    name="get_datetime",
    description="Aktuelles Datum/Uhrzeit.",
    parameters={"format": {"type": "string", "description": "'date', 'time', oder 'full'"}}
)
def get_datetime(format: str = "full") -> str:
    now = datetime.now()
    return {
        "date": now.strftime("%d.%m.%Y"),
        "time": now.strftime("%H:%M:%S"),
        "full": now.strftime("%d.%m.%Y %H:%M:%S")
    }.get(format, now.strftime("%d.%m.%Y %H:%M:%S"))


class ProductionAgent:
    def __init__(self, name: str = "Agent", max_steps: int = 10):
        self.name = name
        self.max_steps = max_steps
        self.conversation = []
        self.total_tokens = 0
        self.total_cost_usd = 0.0

        # OpenAI: System Prompt als role="system" Message
        # Anthropic hatte separaten "system" Parameter im API Call
        self.system_message = {
            "role": "system",
            "content": f"Du bist {name}, ein kompetenter AI Agent. Heute: {datetime.now().strftime('%d.%m.%Y')}. Nutze Tools wenn nötig."
        }

    def _track_cost(self, prompt_tokens: int, completion_tokens: int):
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = (prompt_tokens * 0.00015 + completion_tokens * 0.0006) / 1000
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_cost_usd += cost

    def chat(self, user_message: str) -> str:
        self.conversation.append({"role": "user", "content": user_message})
        print(f"\n[{self.name}] Denke nach...\n")

        for step in range(self.max_steps):
            # System Message immer als erste Message
            messages = [self.system_message] + self.conversation

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                tools=tools.schemas,
                messages=messages
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            self._track_cost(response.usage.prompt_tokens, response.usage.completion_tokens)

            # Tool aufrufen
            if finish_reason == "tool_calls":
                # Assistenten-Message zur History (wichtig: enthält tool_calls info)
                self.conversation.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in message.tool_calls
                    ]
                })

                for tc in message.tool_calls:
                    tool_input = json.loads(tc.function.arguments)
                    print(f"[Tool] {tc.function.name}({json.dumps(tool_input, ensure_ascii=False)})")
                    result = tools.execute(tc.function.name, tool_input)
                    print(f"[Result] {result}")

                    # OpenAI: role="tool" mit tool_call_id
                    # Anthropic: role="user" mit type="tool_result" in einer Liste
                    self.conversation.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })

            # Fertig
            elif finish_reason == "stop":
                final = message.content or ""
                self.conversation.append({"role": "assistant", "content": final})
                print(f"\n[Stats] {self.total_tokens} Tokens | ${self.total_cost_usd:.4f} USD")
                return final

        return "Maximale Schritte erreicht."

    def reset(self):
        self.conversation = []


if __name__ == "__main__":
    agent = ProductionAgent(name="Jarvis")

    print("="*60 + "\nPRODUCTION AGENT (OpenAI gpt-4o-mini)\n" + "="*60)

    print(agent.chat("Was ist heute für ein Datum und rechne 1337 * 42 aus."))
    print(agent.chat("Speichere Notiz: Titel 'AI Learning', Inhalt 'ReAct, Memory, Multi-Agent gelernt'"))
    print(agent.chat("Was hab ich gerade gespeichert?"))

    print(f"\n[Gesamt] {agent.total_tokens} Tokens | ${agent.total_cost_usd:.4f} USD")
