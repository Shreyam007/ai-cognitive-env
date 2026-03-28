from app.environment import CognitiveEnv
from typing import Tuple, Dict, Any

class MultiFactorGrader:
    def evaluate(self, env: CognitiveEnv) -> Tuple[float, Dict[str, Any]]:
        # 1. Completion Ratio
        total_tasks = len(env.tasks)
        completed = sum(1 for t in env.tasks.values() if t.status == 'completed')
        missed = sum(1 for t in env.tasks.values() if t.status == 'missed')
        
        completion_ratio = completed / max(1, total_tasks)
        
        # 2. Stress Factors
        stresses = [entry['stress'] for entry in env.logger.history] if env.logger.history else [0.0]
        avg_stress = sum(stresses) / len(stresses)
        peak_stress = max(stresses)
        
        stress_score = 1.0
        if avg_stress > 60: stress_score -= 0.3
        if avg_stress > 80: stress_score -= 0.3
        if peak_stress >= 95: stress_score -= 0.2
        stress_score = max(0.0, stress_score)
        
        # 3. Efficiency
        useful_actions = sum(1 for entry in env.logger.history if entry['reward'] > 0)
        total_actions = max(1, len(env.logger.history))
        efficiency = useful_actions / total_actions
        
        # 4. Adaptability (Hard Task hook)
        interruptions_handled = sum(1 for t in env.tasks.values() if "Crisis" in t.title and t.status == 'completed')
        adaptability = 1.0 if interruptions_handled > 0 else (0.5 if "hard" in env.scenario_generator.difficulty else 1.0)
        
        # Final Score
        final_score = (0.4 * completion_ratio) + (0.3 * stress_score) + (0.2 * efficiency) + (0.1 * adaptability)
        
        # Penalties for missed deadlines
        final_score -= (missed * 0.02)
        final_score = max(0.0, min(1.0, final_score))
        
        # Explanation Generation
        explanation = f"Agent completed {completed}/{total_tasks} tasks. "
        if missed > 0:
            explanation += f"Agent missed {missed} deadlines causing severe penalties. "
        if avg_stress > 80:
            explanation += "Average stress was highly elevated, indicating poor break scheduling. "
        elif stress_score > 0.8:
            explanation += "Stress was excellently managed. "
            
        if efficiency < 0.3:
            explanation += "A large number of actions were repetitive or penalizing (low efficiency). "
        
        explanation += f"Final calculated cognitive management score: {final_score:.2f}."

        sub_scores = {
            "completion_ratio": completion_ratio,
            "stress_score": stress_score,
            "efficiency": efficiency,
            "adaptability": adaptability,
            "explanation": explanation
        }
        
        return final_score, sub_scores
