"""
=============================================================================
LEKTION 3: MULTI-AGENT SYSTEME — Das ist wo's richtig geil wird
=============================================================================

Ein einzelner Agent kann vieles. Mehrere spezialisierte Agents zusammen
können ALLES.

Das Prinzip: "Separation of Concerns" — jeder Agent ist Experte in einer Sache.

Patterns:

1. ORCHESTRATOR + WORKER
   Orchestrator plant → gibt Aufgaben an Worker → sammelt Ergebnisse
   Beispiel: CEO-Agent gibt Aufgaben an Research-Agent und Writer-Agent

2. PIPELINE
   Agent A → Agent B → Agent C (sequenziell)
   Jeder verfeinert den Output des vorherigen

3. DEBATE (Adversarial)
   Agent A macht Vorschlag → Agent B kritisiert → Agent A verbessert
   Ergebnis: deutlich bessere Qualität

4. PARALLEL
   Mehrere Agents gleichzeitig (async) → Ergebnisse zusammenführen
   Schneller für unabhängige Subtasks

Hier bauen wir: Orchestrator + spezialisierte Worker
Use Case: Automatisierter Research-Report Generator
=============================================================================
"""

import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()

# =============================================================================
# AGENT BASIS-KLASSE
# Alle Agents erben davon — einheitliches Interface
# =============================================================================

@dataclass
class AgentResult:
    """Standardisierter Output jedes Agents."""
    agent_name: str
    task: str
    result: str
    success: bool
    tokens_used: int = 0


class BaseAgent:
    """
    Basis für alle Agents.
    Jeder Agent hat:
    - einen Namen (für Logging)
    - einen System Prompt (seine "Persönlichkeit" / Expertise)
    - eine run() Methode
    """

    def __init__(self, name: str, system_prompt: str, model: str = "claude-haiku-4-5-20251001"):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model

    def run(self, task: str) -> AgentResult:
        """Führt den Agent mit einer Aufgabe aus."""
        print(f"\n[{self.name}] Task: {task[:80]}...")

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=[{"role": "user", "content": task}]
            )
            result = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens

            print(f"[{self.name}] Fertig ({tokens} Tokens)")

            return AgentResult(
                agent_name=self.name,
                task=task,
                result=result,
                success=True,
                tokens_used=tokens
            )
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                task=task,
                result=f"Fehler: {e}",
                success=False
            )


# =============================================================================
# SPEZIALISIERTE WORKER-AGENTS
# Jeder ist Experte in einem Bereich — durch gezielten System Prompt
# =============================================================================

def create_researcher_agent() -> BaseAgent:
    """
    Recherche-Agent: sammelt und strukturiert Informationen.
    In Production: würde echte Web-Search Tools nutzen.
    """
    return BaseAgent(
        name="Researcher",
        system_prompt="""Du bist ein präziser Research-Analyst.
Deine Aufgabe: Zu jedem Thema die wichtigsten Fakten, Trends und Zahlen liefern.
Format: Strukturierte Bullet Points, keine Floskeln, nur Substanz.
Wenn du etwas nicht weißt: sag es klar."""
    )


def create_critic_agent() -> BaseAgent:
    """
    Kritik-Agent: findet Lücken, Fehler, Verbesserungspotenzial.
    Das ist das 'Adversarial' Element — macht den Output besser.
    """
    return BaseAgent(
        name="Critic",
        system_prompt="""Du bist ein strenger Qualitätsprüfer für AI-generierte Inhalte.
Deine Aufgabe: Finde Schwächen, Lücken, Ungenauigkeiten und Verbesserungspotenzial.
Sei direkt und konkret. Keine falschen Komplimente.
Liste Kritikpunkte als nummerierte Liste."""
    )


def create_writer_agent() -> BaseAgent:
    """
    Writer-Agent: transformiert Rohdaten in lesbaren Content.
    """
    return BaseAgent(
        name="Writer",
        system_prompt="""Du bist ein exzellenter technischer Autor.
Deine Aufgabe: Rohe Informationen in klare, prägnante und interessante Texte umwandeln.
Stil: professionell aber zugänglich. Deutsch. Keine unnötigen Füllwörter."""
    )


def create_orchestrator_agent() -> BaseAgent:
    """
    Orchestrator: plant und koordiniert die anderen Agents.
    Das Gehirn des Systems — denkt in Tasks und Abhängigkeiten.
    """
    return BaseAgent(
        name="Orchestrator",
        system_prompt="""Du bist ein Projekt-Manager für AI-Agent-Systeme.
Deine Aufgabe: Komplexe Anfragen in klare Subtasks zerlegen.
Für jede Anfrage erstelle einen Plan mit klaren Aufgaben für:
- Researcher: Was soll recherchiert werden?
- Writer: Was soll geschrieben werden?
Format: JSON mit keys "researcher_task" und "writer_task"."""
    )


# =============================================================================
# DAS MULTI-AGENT SYSTEM
# Orchestriert den ganzen Flow
# =============================================================================

class ResearchReportSystem:
    """
    Automatisiertes Report-System mit 4 spezialisierten Agents.

    Flow:
    User Request
        ↓
    Orchestrator (plant Tasks)
        ↓
    Researcher (sammelt Infos)
        ↓
    Critic (prüft Qualität)
        ↓
    Writer (erstellt finalen Report)
        ↓
    Final Report
    """

    def __init__(self):
        self.orchestrator = create_orchestrator_agent()
        self.researcher = create_researcher_agent()
        self.critic = create_critic_agent()
        self.writer = create_writer_agent()
        self.total_tokens = 0

    def generate_report(self, topic: str) -> str:
        """
        Hauptmethode: Topic rein, fertiger Report raus.
        Intern arbeiten 4 Agents zusammen.
        """
        print(f"\n{'='*60}")
        print(f"MULTI-AGENT REPORT: {topic}")
        print(f"{'='*60}")

        # SCHRITT 1: Orchestrator plant
        # Er zerlegt die Anfrage in konkrete Tasks für jeden Worker
        plan_result = self.orchestrator.run(
            f"Erstelle einen Aufgabenplan für einen Report über: {topic}"
        )
        self.total_tokens += plan_result.tokens_used

        # Plan parsen — in Production: robusteres JSON-Parsing
        import json, re
        try:
            json_match = re.search(r'\{.*\}', plan_result.result, re.DOTALL)
            plan = json.loads(json_match.group()) if json_match else {}
        except:
            plan = {}

        researcher_task = plan.get("researcher_task", f"Recherchiere die wichtigsten Fakten über: {topic}")
        writer_task = plan.get("writer_task", f"Schreibe einen Report über: {topic}")

        # SCHRITT 2: Researcher sammelt Informationen
        research_result = self.researcher.run(researcher_task)
        self.total_tokens += research_result.tokens_used

        # SCHRITT 3: Critic prüft die Recherche
        # Gibt dem Writer Input was noch fehlt oder verbessert werden soll
        critic_result = self.critic.run(
            f"Prüfe diese Recherche auf Vollständigkeit und Qualität:\n\n{research_result.result}"
        )
        self.total_tokens += critic_result.tokens_used

        # SCHRITT 4: Writer erstellt finalen Report
        # Bekommt Recherche + Kritik als Input — berücksichtigt beides
        final_result = self.writer.run(
            f"""{writer_task}

Nutze diese Recherche als Grundlage:
{research_result.result}

Berücksichtige diese Kritikpunkte:
{critic_result.result}

Erstelle einen strukturierten, lesbaren Report."""
        )
        self.total_tokens += final_result.tokens_used

        print(f"\n[System] Gesamt: {self.total_tokens} Tokens genutzt")

        return final_result.result


# =============================================================================
# BONUS: DEBATE PATTERN
# Zwei Agents streiten — Ergebnis ist besser als einer allein
# =============================================================================

def debate(question: str, rounds: int = 2) -> str:
    """
    Zwei Agents debattieren eine Frage.
    Agent A argumentiert PRO, Agent B dagegen.
    Nach N Runden: Zusammenfassung der besten Argumente.
    """
    pro_agent = BaseAgent(
        name="Pro-Agent",
        system_prompt="Du argumentierst IMMER für die These. Finde die stärksten Pro-Argumente."
    )
    contra_agent = BaseAgent(
        name="Contra-Agent",
        system_prompt="Du argumentierst IMMER gegen die These. Finde die stärksten Contra-Argumente."
    )
    summary_agent = BaseAgent(
        name="Moderator",
        system_prompt="Du fasst Debatten neutral zusammen und extrahierst die wichtigsten Erkenntnisse."
    )

    print(f"\n=== DEBATE: {question} ===")
    debate_log = []

    for round_num in range(rounds):
        context = "\n".join(debate_log) if debate_log else ""
        task = f"These: {question}\n\nBisherige Debatte:\n{context}" if context else f"These: {question}"

        pro = pro_agent.run(task)
        contra = contra_agent.run(task)

        debate_log.append(f"RUNDE {round_num+1} PRO:\n{pro.result}")
        debate_log.append(f"RUNDE {round_num+1} CONTRA:\n{contra.result}")

    summary = summary_agent.run(
        f"Fasse diese Debatte zusammen:\n\n" + "\n\n".join(debate_log)
    )
    return summary.result


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    system = ResearchReportSystem()

    # Multi-Agent Report
    report = system.generate_report("Die Zukunft von AI Agents in der Softwareentwicklung")
    print("\n" + "="*60)
    print("FINALER REPORT:")
    print("="*60)
    print(report)

    # Debate Pattern
    result = debate("AI Agents werden Software-Entwickler in 5 Jahren ersetzen", rounds=1)
    print("\n=== DEBATE ERGEBNIS ===")
    print(result)
