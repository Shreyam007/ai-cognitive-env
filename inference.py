import os
import sys
import json
import io
import contextlib
from app.environment import CognitiveEnv
from app.agents import BaselineAgent
from app.tasks import ScenarioGenerator
from app.grader import MultiFactorGrader

def run_task(task_name):
    # 1. Environment and Scenario Setup
    env = CognitiveEnv()
    scenario = ScenarioGenerator(task_name)
    
    # 2. Reset Environment
    obs = env.reset(seed=42, scenario_generator=scenario)
    
    # 3. [START] log
    print(f"[START] task={task_name}", flush=True)
    
    agent = BaselineAgent()
    grader = MultiFactorGrader()
    done = False
    step_count = 0
    max_steps = 50
    
    while not done and step_count < max_steps:
        step_count += 1
        
        # Agent Decision - Suppress potential debug prints from agent to stdout
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            action = agent.decide(obs)
        
        # Environment Step
        obs, reward, done, info = env.step(action)
        
        # 4. [STEP] log
        print(f"[STEP] step={step_count} reward={reward:.2f}", flush=True)
        
    # 5. [END] log
    final_score, _ = grader.evaluate(env)
    print(f"[END] task={task_name} score={final_score:.2f} steps={step_count}", flush=True)

def main():
    # Implement for ALL tasks: easy, medium, hard
    for task in ["easy", "medium", "hard"]:
        run_task(task)

if __name__ == "__main__":
    main()
