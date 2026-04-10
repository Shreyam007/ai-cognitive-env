from app.environment import CognitiveEnv
from typing import Tuple, Dict, Any

class MultiFactorGrader:

    def evaluate(self, env: CognitiveEnv) -> Tuple[int, Dict[str, Any]]:
        total_tasks = len(env.tasks)
        completed = sum(1 for t in env.tasks.values() if t.status == "completed")
        missed = sum(1 for t in env.tasks.values() if t.status == "missed")

        history = getattr(env.logger, "history", [])
        stresses = [entry.get("stress", 50) for entry in history] or [50]

        avg_stress = sum(stresses) / len(stresses)

        # Raw metrics evaluated to strict integers 0 or 1
        completion_ratio = 1 if (completed / max(1, total_tasks)) > 0.5 else 0
        
        useful_actions = sum(1 for entry in history if entry.get("reward", 0) > 0)
        total_actions = max(1, len(history))
        efficiency = 1 if (useful_actions / total_actions) > 0.5 else 0

        interruptions_handled = sum(
            1 for t in env.tasks.values()
            if "Crisis" in getattr(t, "title", "") and t.status == "completed"
        )

        adaptability = 1 if interruptions_handled > 0 else 0

        # Stress score logic to integer 0 or 1
        stress_score = 1 if avg_stress <= 60 else 0

        # Strict integer calculation for final score (0 or 1)
        # Any penalty (missed > 0) or bad ratios will flip to 0
        if missed > 0 or efficiency == 0 or stress_score == 0:
            final_score = 0
        else:
            final_score = 1 if completion_ratio == 1 else 0

        explanation = f"Agent completed {completed}/{total_tasks} tasks. "
        if missed > 0:
            explanation += f"Missed {missed} deadlines causing total failure. "
        if stress_score == 0:
            explanation += "Average stress was excessively high. "
        elif stress_score == 1:
            explanation += "Stress was excellently managed. "

        if efficiency == 0:
            explanation += "Efficiency was low due to repetitive actions. "

        # EXACT USER DEMAND: NO DECIMALS, ONLY 0 OR 1
        explanation += f"Final calculated cognitive management score: {final_score}."

        sub_scores = {
            "completion_ratio": completion_ratio,
            "stress_score": stress_score,
            "efficiency": efficiency,
            "adaptability": adaptability,
            "explanation": explanation
        }

        return int(final_score), sub_scores
