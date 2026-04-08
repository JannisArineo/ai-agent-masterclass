"""
=============================================================================
LEKTION 4: SELF-REFLECTION — Agents die sich selbst verbessern
=============================================================================

Das ist das was AI Agents von normalen LLM-Calls unterscheidet.

Reflection bedeutet: der Agent bewertet seinen eigenen Output und
verbessert ihn iterativ — ohne dass ein Mensch eingreifen muss.

Patterns:

1. SELF-CRITIQUE
   Agent generiert Output → bewertet ihn → verbessert ihn

2. REFLEXION (Shinn et al. 2023)
   Agent scheitert → analysiert warum → speichert Learnings →
   nächster Versuch nutzt die Learnings

3. CONSTITUTIONAL AI
   Agent prüft seinen Output gegen Regeln ("Constitution")
   Ist es sicher? Ist es korrekt? Ist es hilfreich?

4. TREE OF THOUGHTS
   Agent exploriert mehrere Lösungspfade gleichzeitig,
   bewertet sie und verfolgt den besten weiter

Hier implementieren wir: Self-Critique + Reflexion
=============================================================================
"""

import anthropic
from dataclasses import dataclass, field

client = anthropic.Anthropic()


# =============================================================================
# SELF-CRITIQUE LOOP
# Generate → Critique → Improve → Repeat
# =============================================================================

def self_critique_loop(task: str, max_iterations: int = 3, target_score: int = 8) -> str:
    """
    Verbessert einen Output durch iterative Selbstkritik.

    Ablauf:
    1. Ersten Entwurf erstellen
    2. Entwurf selbst bewerten (Score 1-10 + Begründung)
    3. Wenn Score < target_score: verbessern → zurück zu 2
    4. Wenn Score >= target_score oder max_iterations erreicht: fertig

    Das ist wie ein Mensch der seinen eigenen Aufsatz mehrfach überarbeitet.
    """

    print(f"\n{'='*60}")
    print(f"SELF-CRITIQUE LOOP: {task[:60]}...")
    print(f"Ziel-Score: {target_score}/10, Max Iterations: {max_iterations}")
    print(f"{'='*60}")

    current_output = None

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # SCHRITT 1: Output generieren (oder verbessern)
        if current_output is None:
            # Erster Entwurf
            prompt = f"Führe folgende Aufgabe aus:\n{task}"
        else:
            # Verbesserung basierend auf Kritik
            prompt = f"""Verbessere diesen Output basierend auf der Kritik.

Aufgabe: {task}

Aktueller Output:
{current_output['text']}

Kritik:
{current_output['critique']}
Score: {current_output['score']}/10

Erstelle eine verbesserte Version die die Kritikpunkte adressiert."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        new_output = response.content[0].text

        # SCHRITT 2: Output bewerten
        critique_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""Bewerte diesen Output für die Aufgabe.

Aufgabe: {task}

Output:
{new_output}

Antworte in diesem Format:
SCORE: [1-10]
STÄRKEN: [was gut ist]
SCHWÄCHEN: [was fehlt oder schlecht ist]
VERBESSERUNG: [konkrete Vorschläge]"""
            }]
        )

        critique_text = critique_response.content[0].text

        # Score extrahieren
        import re
        score_match = re.search(r'SCORE:\s*(\d+)', critique_text)
        score = int(score_match.group(1)) if score_match else 5

        current_output = {
            "text": new_output,
            "critique": critique_text,
            "score": score,
            "iteration": iteration + 1
        }

        print(f"Score: {score}/10")
        print(f"Kritik: {critique_text[:200]}...")

        # Ziel erreicht?
        if score >= target_score:
            print(f"\n✓ Ziel-Score erreicht in Iteration {iteration + 1}!")
            break

    print(f"\nFinaler Score: {current_output['score']}/10")
    return current_output["text"]


# =============================================================================
# REFLEXION AGENT
# Lernt aus Fehlern — speichert "Learnings" und nutzt sie beim nächsten Versuch
# =============================================================================

@dataclass
class Reflection:
    """Ein gespeichertes Learning aus einem fehlgeschlagenen Versuch."""
    task_type: str
    failure: str
    learning: str
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())


class ReflexionAgent:
    """
    Agent der aus Fehlern lernt.

    Der Reflexion-Mechanismus (Shinn et al. 2023):
    1. Versuch mit aktuellem Wissen
    2. Bewertung: Hat es funktioniert?
    3. Falls nicht: Analysiere warum → schreibe Reflection
    4. Speichere Reflection in Memory
    5. Nächster Versuch startet mit Reflections als Kontext

    Das ist wie ein Mensch der nach einem gescheiterten Projekt analysiert
    was schief gelaufen ist und es beim nächsten Mal besser macht.
    """

    def __init__(self):
        self.reflections: list[Reflection] = []  # Langzeit-Learnings
        self.max_attempts = 3

    def _get_reflection_context(self, task_type: str) -> str:
        """Holt relevante Learnings aus vergangenen Fehlern."""
        relevant = [r for r in self.reflections if r.task_type == task_type]
        if not relevant:
            return "Keine Vorerfahrung mit diesem Aufgabentyp."

        context = "Learnings aus vergangenen Versuchen:\n"
        for r in relevant[-3:]:  # Letzte 3 Reflections
            context += f"- Fehler: {r.failure}\n  Learning: {r.learning}\n"
        return context

    def _evaluate(self, task: str, output: str, criteria: str) -> tuple[bool, str]:
        """
        Bewertet ob ein Output die Aufgabe erfüllt.
        Gibt (success, feedback) zurück.
        """
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"""Bewerte ob dieser Output die Aufgabe korrekt erfüllt.

Aufgabe: {task}
Kriterien: {criteria}
Output: {output}

Antworte NUR mit:
ERFOLG: ja/nein
FEEDBACK: [warum erfolgreich/was fehlt]"""
            }]
        )

        text = response.content[0].text
        success = "erfolg: ja" in text.lower()
        feedback_match = re.search(r'FEEDBACK:\s*(.+)', text, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else text

        return success, feedback

    def _reflect(self, task: str, task_type: str, attempt: str, failure_reason: str):
        """
        Analysiert warum ein Versuch gescheitert ist und extrahiert ein Learning.
        """
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"""Ein Agent hat versucht eine Aufgabe zu lösen und ist gescheitert.
Analysiere den Fehler und formuliere ein konkretes Learning für zukünftige Versuche.

Aufgabe: {task}
Versuch: {attempt[:200]}
Fehlergrund: {failure_reason}

Formuliere das Learning in 1-2 Sätzen, konkret und actionable."""
            }]
        )

        learning = response.content[0].text.strip()
        reflection = Reflection(
            task_type=task_type,
            failure=failure_reason[:100],
            learning=learning
        )
        self.reflections.append(reflection)
        print(f"[Reflexion] Neues Learning: {learning[:100]}")

    def solve(self, task: str, task_type: str, criteria: str) -> str:
        """
        Löst eine Aufgabe mit Reflexion bei Misserfolg.
        """
        import re
        print(f"\n{'='*60}")
        print(f"REFLEXION AGENT: {task[:60]}...")
        print(f"{'='*60}")

        for attempt_num in range(self.max_attempts):
            print(f"\n--- Versuch {attempt_num + 1}/{self.max_attempts} ---")

            # Relevante Learnings aus der Vergangenheit holen
            reflection_context = self._get_reflection_context(task_type)

            # Aufgabe lösen — mit Reflexions-Kontext
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": f"""{reflection_context}

Aufgabe: {task}

Löse die Aufgabe sorgfältig."""
                }]
            )

            output = response.content[0].text
            print(f"Output: {output[:200]}...")

            # Bewertung
            success, feedback = self._evaluate(task, output, criteria)
            print(f"Erfolg: {success}, Feedback: {feedback[:100]}")

            if success:
                print(f"\n✓ Erfolgreich in Versuch {attempt_num + 1}!")
                return output

            # Gescheitert — Reflexion erstellen
            if attempt_num < self.max_attempts - 1:
                self._reflect(task, task_type, output, feedback)

        print("\n⚠ Maximale Versuche erreicht.")
        return output  # Besten letzten Versuch zurückgeben


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    import re

    print("\n=== DEMO 1: Self-Critique Loop ===")
    result = self_critique_loop(
        task="Erkläre AI Agents einem 10-jährigen Kind in 3 Sätzen.",
        max_iterations=2,
        target_score=8
    )
    print("\nFinales Ergebnis:")
    print(result)

    print("\n\n=== DEMO 2: Reflexion Agent ===")
    agent = ReflexionAgent()

    # Aufgabe die Präzision erfordert
    result = agent.solve(
        task="Schreibe exakt 3 Fakten über Python als nummerierte Liste. Nur Fakten, kein Intro.",
        task_type="structured_list",
        criteria="Muss exakt 3 nummerierte Punkte haben, nur Fakten, keine Einleitung"
    )
    print("\nFinales Ergebnis:")
    print(result)
