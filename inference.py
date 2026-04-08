import os
import sys
import json
import io
import contextlib
import traceback

# Move imports inside try block to handle environment issues gracefully
def run_task(task_name):
    # 1. Start Log
    print(f"[START] task={task_name}", flush=True)
    
    try:
        from app.environment import CognitiveEnv
        from app.agents import LLMAgent
        from app.tasks import ScenarioGenerator
        from app.grader import MultiFactorGrader

        env = CognitiveEnv()
        scenario = ScenarioGenerator(task_name)
        obs = env.reset(seed=42, scenario_generator=scenario)
        
        agent = LLMAgent()
        grader = MultiFactorGrader()
        done = False
        step_count = 0
        
        while not done:
            step_count += 1
            # MUST call agent.act() which triggers LLM
            action = agent.act(obs)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # 2. Step Log
            print(f"[STEP] step={step_count} reward={reward:.2f}", flush=True)

        # 3. End Log
        final_score, _ = grader.evaluate(env)
        print(f"[END] task={task_name} score={final_score:.2f} steps={step_count}", flush=True)

    except Exception as e:
        import traceback
        print(f"CRITICAL FAILURE: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Ensure [END] is still printed even on failure to avoid hanging validation
        print(f"[END] task={task_name} score=0.0 steps=0", flush=True)

def main():
    # Only run 'easy' for high speed validation if needed, or all 3
    for task in ["easy"]:
        run_task(task)

if __name__ == "__main__":
    main()
