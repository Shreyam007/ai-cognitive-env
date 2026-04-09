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
            # Real API call triggered here via LLMAgent.act()
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            
            # Use raw reward formatting to match exact validator expectations
            print(f"[STEP] step={step_count} reward={reward}", flush=True)

        final_score, _ = grader.evaluate(env)
        print(f"[END] task={task_name} score={final_score} steps={step_count}", flush=True)

    except (ImportError, ModuleNotFoundError) as e:
        import traceback
        print(f"FAILED TO START: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # We only catch imports to avoid environment-setup blockers, 
        # but simulation errors should crash.
        print(f"[END] task={task_name} score=0.0 steps=0", flush=True)

def main():
    # Only run 'easy' for high speed validation if needed, or all 3
    for task in ["easy"]:
        run_task(task)

if __name__ == "__main__":
    main()
