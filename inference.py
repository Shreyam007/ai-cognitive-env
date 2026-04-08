import os
import sys
import json
import io
import contextlib
import traceback

# Move imports inside try block to handle environment issues gracefully
def run_task(task_name):
    # 1. Immediate [START] log to satisfy validator
    print(f"[START] task={task_name}", flush=True)
    sys.stdout.flush()
    
    try:
        from app.environment import CognitiveEnv
        from app.agents import BaselineAgent
        from app.tasks import ScenarioGenerator
        from app.grader import MultiFactorGrader

        # 2. Environment and Scenario Setup
        env = CognitiveEnv()
        scenario = ScenarioGenerator(task_name)
        
        # 3. Reset Environment
        obs = env.reset(seed=42, scenario_generator=scenario)
        
        agent = BaselineAgent()
        # Diagnostic log to stderr
        client_status = "READY" if agent.client else "MISSING (FALLBACK TO RULE-BASED)"
        print(f"DEBUG: Agent initialized. LLM Client: {client_status}", file=sys.stderr)
        
        grader = MultiFactorGrader()
        done = False
        step_count = 0
        max_steps = 50
        
        while not done and step_count < max_steps:
            step_count += 1
            
            # Agent Decision
            action = agent.decide(obs)
            
            # Environment Step
            obs, reward, done, info = env.step(action)
            
            # 4. [STEP] log
            print(f"[STEP] step={step_count} reward={reward:.2f}", flush=True)
            sys.stdout.flush()
            
        # 5. [END] log
        final_score, _ = grader.evaluate(env)
        print(f"[END] task={task_name} score={final_score:.2f} steps={step_count}", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"ERROR running task {task_name}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Still print an [END] block if START was printed, to maintain structure, 
        # but with zero score if it failed catastrophically.
        print(f"[END] task={task_name} score=0.0 steps=0", flush=True)
        sys.stdout.flush()

def main():
    # Implement for ALL tasks: easy, medium, hard
    for task in ["easy", "medium", "hard"]:
        run_task(task)

if __name__ == "__main__":
    main()
