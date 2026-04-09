from app.environment import CognitiveEnv
from typing import Tuple, Dict, Any

class MultiFactorGrader:
    def _clamp_open_01(self, value: float) -> float:
        value = float(value)
        if value <= 0.0:
            return 0.01
        if value >= 1.0:
            return 0.99
        return value

    def _band_score(self, value: float) -> float:
        """
        Convert any numeric value into a safe strict-open interval score.
        """
        if value <= 0.0:
            return 0.01
        if value >= 1.0:
            return 0.99
        return round(max(0.01, min(0.99, value)), 2)

    def evaluate(self, env: CognitiveEnv) -> Tuple[float, Dict[str, Any]]:
        total_tasks = len(env.tasks)
        completed = sum(1 for t in env.tasks.values() if t.status == "completed")
        missed = sum(1 for t in env.tasks.values() if t.status == "missed")

        history = getattr(env.logger, "history", [])
        stresses = [entry.get("stress", 0.5) for entry in history]
        if not stresses:
            stresses = [0.5]

        avg_stress = sum(stresses) / len(stresses)
        peak_stress = max(stresses)

        # Base metrics
        completion_ratio = completed / max(1, total_tasks)
        useful_actions = sum(1 for entry in history if entry.get("reward", 0) > 0)
        total_actions = max(1, len(history))
        efficiency = useful_actions / total_actions

        interruptions_handled = sum(
            1 for t in env.tasks.values()
            if "Crisis" in getattr(t, "title", "") and t.status == "completed"
        )

        difficulty = getattr(env.scenario_generator, "difficulty", "")
        if interruptions_handled > 0:
            adaptability = 0.90
        else:
            adaptability = 0.65 if "hard" in difficulty else 0.80

        # Stress score from stress levels
        stress_score = 0.95
        if avg_stress > 60:
            stress_score -= 0.25
        if avg_stress > 80:
            stress_score -= 0.25
        if peak_stress >= 95:
            stress_score -= 0.20

        # Final score
        final_score = (
            0.4 * completion_ratio +
            0.3 * stress_score +
            0.2 * efficiency +
            0.1 * adaptability
        )

        # Penalty for missed tasks
        final_score -= missed * 0.02

        # Safe clamping
        completion_ratio = self._band_score(completion_ratio)
        stress_score = self._band_score(stress_score)
        efficiency = self._band_score(efficiency)
        adaptability = self._band_score(adaptability)
        final_score = self._band_score(final_score)

        explanation = f"Agent completed {completed}/{total_tasks} tasks."
        if missed > 0:
            explanation += f" Missed {missed} deadlines."
        if avg_stress > 80:
            explanation += " Average stress was high."
        elif avg_stress < 40:
            explanation += " Average stress was well controlled."
        if efficiency < 0.3:
            explanation += " Efficiency was low due to repetitive actions."
        explanation += f" Final score: {final_score:.2f}."

        sub_scores = {
            "completion_ratio": completion_ratio,
            "stress_score": stress_score,
            "efficiency": efficiency,
            "adaptability": adaptability,
            "explanation": explanation
        }

        return final_score, sub_scores
