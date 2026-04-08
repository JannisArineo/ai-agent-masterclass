"""
=============================================================================
LEKTION 4: SELF-REFLECTION (OpenAI Version)
=============================================================================
"""

import re
from dataclasses import dataclass, field
from openai import OpenAI

client = OpenAI()


def self_critique_loop(task: str, max_iterations: int = 3, target_score: int = 8) -> str:
    print(f"\n{'='*60}\nSELF-CRITIQUE: {task[:60]}...\nZiel: {target_score}/10\n{'='*60}")

    current_output = None

    for i in range(max_iterations):
        print(f"\n--- Iteration {i + 1} ---")

        if current_output is None:
            prompt = f"Führe aus:\n{task}"
        else:
            prompt = f"""Verbessere den Output basierend auf der Kritik.

Aufgabe: {task}
Aktueller Output: {current_output['text']}
Kritik: {current_output['critique']}
Score: {current_output['score']}/10"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        new_output = response.choices[0].message.content

        critique_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""Bewerte für Aufgabe: {task}

Output: {new_output}

Format:
SCORE: [1-10]
STÄRKEN: ...
SCHWÄCHEN: ...
VERBESSERUNG: ..."""}]
        )

        critique_text = critique_response.choices[0].message.content
        score_match = re.search(r'SCORE:\s*(\d+)', critique_text)
        score = int(score_match.group(1)) if score_match else 5

        current_output = {"text": new_output, "critique": critique_text, "score": score}
        print(f"Score: {score}/10")

        if score >= target_score:
            print(f"✓ Ziel erreicht!")
            break

    return current_output["text"]


@dataclass
class Reflection:
    task_type: str
    failure: str
    learning: str


class ReflexionAgent:
    def __init__(self):
        self.reflections: list[Reflection] = []
        self.max_attempts = 3

    def _get_context(self, task_type: str) -> str:
        relevant = [r for r in self.reflections if r.task_type == task_type]
        if not relevant:
            return "Keine Vorerfahrung."
        return "Learnings:\n" + "\n".join(f"- {r.learning}" for r in relevant[-3:])

    def _evaluate(self, task: str, output: str, criteria: str) -> tuple[bool, str]:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""Erfüllt der Output die Aufgabe?

Aufgabe: {task}
Kriterien: {criteria}
Output: {output}

NUR:
ERFOLG: ja/nein
FEEDBACK: ..."""}]
        )
        text = response.choices[0].message.content
        success = "erfolg: ja" in text.lower()
        feedback = re.search(r'FEEDBACK:\s*(.+)', text, re.DOTALL)
        return success, (feedback.group(1).strip() if feedback else text)

    def _reflect(self, task: str, task_type: str, attempt: str, failure: str):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""Agent ist gescheitert. Formuliere ein Learning.

Aufgabe: {task}
Versuch: {attempt[:200]}
Fehler: {failure}

1-2 Sätze, konkret."""}]
        )
        learning = response.choices[0].message.content.strip()
        self.reflections.append(Reflection(task_type=task_type, failure=failure[:100], learning=learning))
        print(f"[Reflexion] {learning[:100]}")

    def solve(self, task: str, task_type: str, criteria: str) -> str:
        print(f"\n{'='*60}\nREFLEXION AGENT: {task[:60]}...\n{'='*60}")

        for attempt_num in range(self.max_attempts):
            print(f"\n--- Versuch {attempt_num + 1} ---")
            context = self._get_context(task_type)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"{context}\n\nAufgabe: {task}"}]
            )
            output = response.choices[0].message.content
            print(f"Output: {output[:200]}...")

            success, feedback = self._evaluate(task, output, criteria)
            print(f"Erfolg: {success}")

            if success:
                print(f"✓ Fertig in Versuch {attempt_num + 1}!")
                return output

            if attempt_num < self.max_attempts - 1:
                self._reflect(task, task_type, output, feedback)

        return output


if __name__ == "__main__":
    result = self_critique_loop("Erkläre AI Agents einem 10-jährigen in 3 Sätzen.", max_iterations=2)
    print("\nErgebnis:", result)
