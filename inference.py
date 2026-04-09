import os
import sys
import json
import io
import contextlib
import traceback

# Move imports inside try block to handle environment issues gracefully
def run_task(task_name):
    # This will raise KeyError immediately if MODEL_NAME is missing
    model_name = os.environ["MODEL_NAME"]
    # Exact start format
    print(f"[START] task={task_name} env=CognitiveLoadEnv model={model_name}", flush=True)
    
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
    all_rewards = []
    
    while not done:
        step_count += 1
        # Forced LLM call
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        all_rewards.append(reward)
        
        # Exact step format: done is lowercase true/false
        done_str = "true" if done else "false"
        print(f"[STEP] step={step_count} action={action.action_type} reward={reward} done={done_str} error=null", flush=True)

    final_score, _ = grader.evaluate(env)
    # Exact end format: success is true if score > 0
    success_str = "true" if final_score > 0 else "false"
    rewards_str = ",".join([str(r) for r in all_rewards])
    print(f"[END] success={success_str} steps={step_count} score={final_score} rewards={rewards_str}", flush=True)

def main():
    # Run all tasks as required by the full spec
    for task in ["easy", "medium", "hard"]:
        try:
            run_task(task)
        except Exception as e:
            # If a task fails, raise it to stderr so the platform knows the execution failed
            import sys
            print(f"ERROR in task {task}: {e}", file=sys.stderr)
            raise

if __name__ == "__main__":
    main()
