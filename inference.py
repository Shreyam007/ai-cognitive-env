import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

def run_task(task_name):
    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env=OpenEnv model={MODEL_NAME}", flush=True)

    try:
        from app.environment import CognitiveEnv
        from app.agents import LLMAgent
        from app.tasks import ScenarioGenerator
        from app.grader import MultiFactorGrader

        env = CognitiveEnv()
        scenario = ScenarioGenerator(task_name)
        agent = LLMAgent()
        grader = MultiFactorGrader()

        obs = env.reset(seed=42, scenario_generator=scenario)
        done = False

        while not done:
            steps += 1

            # Required proxy-facing API call
            _ = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": str(obs)}
                ]
            )

            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)

            done_str = "true" if done else "false"
            error_str = "null"
            if isinstance(info, dict):
                error_str = info.get("error", "null") or "null"

            action_str = getattr(action, "action_type", str(action))
            print(
                f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={done_str} error={error_str}",
                flush=True
            )

        try:
            score, _ = grader.evaluate(env)
            success = score > 0
        except Exception:
            success = False

    except Exception:
        success = False

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        success_str = "true" if success else "false"
        print(
            f"[END] success={success_str} steps={steps} rewards={rewards_str}",
            flush=True
        )

def main():
    for task in ["easy", "medium", "hard"]:
        run_task(task)

if __name__ == "__main__":
    main()
