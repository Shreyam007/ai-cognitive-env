from fastapi import FastAPI, BackgroundTasks
from app.models import Observation, Action
from app.environment import CognitiveEnv
from app.tasks import ScenarioGenerator
from app.grader import MultiFactorGrader
from app.agents import BaselineAgent

app = FastAPI(
    title="AI Cognitive Load Management Environment",
    description="An OpenEnv-compatible evaluation system exposing dynamic stress, energy, and priority mechanics.",
    version="1.0.0"
)

# Shared environment instance for simplified API testing constraints
env = CognitiveEnv()

@app.get("/tasks")
def list_tasks():
    return {
        "observation_schema": Observation.model_json_schema(),
        "action_schema": Action.model_json_schema(),
        "current_active_tasks": env.state().get("tasks")
    }

@app.get("/grader")
def run_grader():
    grader = MultiFactorGrader()
    score, breakdown = grader.evaluate(env)
    return {
        "score": score,
        "breakdown": breakdown
    }

@app.post("/baseline")
def run_baseline(background_tasks: BackgroundTasks, difficulty: str = "hard"):
    # Queue the heavy episodic simulation in the background to avoid endpoint blocking/timeout
    background_tasks.add_task(run_agent_background, difficulty)
    
    return {
        "status": "processing_in_background",
        "message": f"Baseline test started for {difficulty} scenario. Evaluation processing asynchronously."
    }

def run_agent_background(difficulty: str):
    local_env = CognitiveEnv()
    scenario = ScenarioGenerator(difficulty)
    obs = local_env.reset(seed=42, scenario_generator=scenario)
    agent = BaselineAgent()
    done = False
    
    while not done:
        action = agent.decide(obs)
        obs, reward, done, info = local_env.step(action)
        
    grader = MultiFactorGrader()
    score, breakdown = grader.evaluate(local_env)
    local_env.logger.plot_history(filename=f"baseline_{difficulty}_log.png")
    # In production, cache/DB persistence would occur here
