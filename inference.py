import os
import sys
import json
import io
import contextlib
import traceback

# Move imports inside try block to handle environment issues gracefully
def run_task(task_name):
    model_name = os.getenv("MODEL_NAME", "unknown")
    # [START] task=<task_name> env=CognitiveLoadEnv model=<model_name>
    print(f"[START] task={task_name} env=CognitiveLoadEnv model={model_name}", flush=True)
    
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
        all_rewards = []
        
        while not done:
            step_count += 1
            # Real API call triggered here via LLMAgent.act()
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            all_rewards.append(reward)
            
            # [STEP] step=<n> action=<action> reward=<reward> done=<true|false> error=null
            done_str = "true" if done else "false"
            action_str = str(action.action_type)
            print(f"[STEP] step={step_count} action={action_str} reward={reward} done={done_str} error=null", flush=True)

        final_score, _ = grader.evaluate(env)
        # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
        success_str = "true" if final_score > 0 else "false"
        rewards_str = ",".join([str(r) for r in all_rewards])
        print(f"[END] success={success_str} steps={step_count} score={final_score} rewards={rewards_str}", flush=True)

    except (ImportError, ModuleNotFoundError) as e:
        import traceback
        print(f"FAILED TO START: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Structured error output
        print(f"[END] success=false steps=0 score=0.0 rewards=[]", flush=True)

def main():
    # Only run 'easy' for high speed validation if needed, or all 3
    for task in ["easy"]:
        run_task(task)

if __name__ == "__main__":
    main()
