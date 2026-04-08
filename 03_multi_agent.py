"""
=============================================================================
LEKTION 3: MULTI-AGENT SYSTEME (OpenAI Version)
=============================================================================

OpenAI Besonderheit:
- System Prompt als role="system" Message (nicht separater Parameter)
- Ansonsten gleiche Patterns wie Anthropic
=============================================================================
"""

import json
import re
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI()


@dataclass
class AgentResult:
    agent_name: str
    task: str
    result: str
    success: bool
    tokens_used: int = 0


class BaseAgent:
    def __init__(self, name: str, system_prompt: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model

    def run(self, task: str) -> AgentResult:
        print(f"\n[{self.name}] Task: {task[:80]}...")
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},  # OpenAI: system als Message
                    {"role": "user", "content": task}
                ]
            )
            result = response.choices[0].message.content
            tokens = response.usage.prompt_tokens + response.usage.completion_tokens  # OpenAI: prompt_tokens statt input_tokens
            print(f"[{self.name}] Fertig ({tokens} Tokens)")
            return AgentResult(agent_name=self.name, task=task, result=result, success=True, tokens_used=tokens)
        except Exception as e:
            return AgentResult(agent_name=self.name, task=task, result=f"Fehler: {e}", success=False)


def create_researcher_agent() -> BaseAgent:
    return BaseAgent(
        name="Researcher",
        system_prompt="Du bist ein präziser Research-Analyst. Liefere Fakten, Trends, Zahlen als Bullet Points. Kein Fülltext."
    )

def create_critic_agent() -> BaseAgent:
    return BaseAgent(
        name="Critic",
        system_prompt="Du prüfst AI-generierten Content auf Schwächen und Lücken. Direkt, konkret, nummerierte Liste."
    )

def create_writer_agent() -> BaseAgent:
    return BaseAgent(
        name="Writer",
        system_prompt="Du verwandelst Rohdaten in klare, prägnante Texte. Professionell, Deutsch, keine Füllwörter."
    )

def create_orchestrator_agent() -> BaseAgent:
    return BaseAgent(
        name="Orchestrator",
        system_prompt="""Du zerlegst Anfragen in Tasks für Researcher und Writer.
Antworte NUR mit JSON: {"researcher_task": "...", "writer_task": "..."}"""
    )


class ResearchReportSystem:
    """
    Flow: Orchestrator → Researcher → Critic → Writer → Report
    """
    def __init__(self):
        self.orchestrator = create_orchestrator_agent()
        self.researcher = create_researcher_agent()
        self.critic = create_critic_agent()
        self.writer = create_writer_agent()
        self.total_tokens = 0

    def generate_report(self, topic: str) -> str:
        print(f"\n{'='*60}\nMULTI-AGENT REPORT: {topic}\n{'='*60}")

        # Plan
        plan_result = self.orchestrator.run(f"Erstelle Aufgabenplan für Report über: {topic}")
        self.total_tokens += plan_result.tokens_used
        try:
            json_match = re.search(r'\{.*\}', plan_result.result, re.DOTALL)
            plan = json.loads(json_match.group()) if json_match else {}
        except:
            plan = {}

        researcher_task = plan.get("researcher_task", f"Recherchiere Fakten über: {topic}")
        writer_task = plan.get("writer_task", f"Schreibe Report über: {topic}")

        # Research
        research = self.researcher.run(researcher_task)
        self.total_tokens += research.tokens_used

        # Kritik
        critique = self.critic.run(f"Prüfe diese Recherche:\n\n{research.result}")
        self.total_tokens += critique.tokens_used

        # Finaler Report
        final = self.writer.run(
            f"{writer_task}\n\nRecherche:\n{research.result}\n\nKritikpunkte:\n{critique.result}"
        )
        self.total_tokens += final.tokens_used

        print(f"\n[System] Gesamt: {self.total_tokens} Tokens")
        return final.result


def debate(question: str, rounds: int = 2) -> str:
    pro = BaseAgent("Pro-Agent", "Du argumentierst IMMER für die These. Stärkste Pro-Argumente.")
    contra = BaseAgent("Contra-Agent", "Du argumentierst IMMER gegen die These. Stärkste Contra-Argumente.")
    moderator = BaseAgent("Moderator", "Du fasst Debatten neutral zusammen.")

    print(f"\n=== DEBATE: {question} ===")
    log = []

    for r in range(rounds):
        context = "\n".join(log) if log else ""
        task = f"These: {question}\n\nBisherige Debatte:\n{context}" if context else f"These: {question}"
        log.append(f"RUNDE {r+1} PRO:\n{pro.run(task).result}")
        log.append(f"RUNDE {r+1} CONTRA:\n{contra.run(task).result}")

    return moderator.run("Fasse diese Debatte zusammen:\n\n" + "\n\n".join(log)).result


if __name__ == "__main__":
    system = ResearchReportSystem()
    report = system.generate_report("Die Zukunft von AI Agents in der Softwareentwicklung")
    print("\n" + "="*60 + "\nFINALER REPORT:\n" + "="*60)
    print(report)
