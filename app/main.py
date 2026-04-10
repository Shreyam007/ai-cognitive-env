from fastapi import FastAPI, BackgroundTasks
from app.models import Observation, Action
from app.environment import CognitiveEnv
from app.tasks import ScenarioGenerator
from app.grader import MultiFactorGrader
from app.agents import LLMAgent

def clamp_score(value):
    """Force any numeric value strictly into (0, 1) open interval."""
    value = round(float(value), 4)
    if value <= 0.0:
        return 0.01
    if value >= 1.0:
        return 0.99
    return value

app = FastAPI(
    title="AI Cognitive Load Management Environment",
    description="An OpenEnv-compatible evaluation system exposing dynamic stress, energy, and priority mechanics.",
    version="1.0.0"
)

# Shared environment instance for simplified API testing constraints
env = CognitiveEnv()

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "AI Cognitive Load Environment is live and OpenEnv compliant.",
        "version": "1.0.0"
    }

@app.get("/tasks")
def list_tasks():
    return {
        "observation_schema": Observation.model_json_schema(),
        "action_schema": Action.model_json_schema(),
        "current_active_tasks": env.state().get("tasks")
    }

from fastapi import Request

@app.post("/reset")
async def reset_env(request: Request):
    try:
        data = await request.json()
    except:
        data = {}
    
    # Handle seed from query param or body
    seed_q = request.query_params.get("seed")
    seed = int(seed_q) if seed_q else data.get("seed")
    
    obs = env.reset(seed=seed)
    return obs  # Return raw observation dict directly

@app.post("/step")
async def step_env(request: Request):
    data = await request.json()
    
    # Standard OpenEnv might send {"action": {...}} or just {...} directly
    action_data = data.get("action", data)
    
    obs, reward, done, info = env.step(action_data)
    return {
        "observation": obs,
        "reward": clamp_score(reward),
        "done": bool(done),
        "info": info
    }

@app.get("/state")
def get_state():
    return env.state()

@app.get("/grader")
def run_grader():
    grader = MultiFactorGrader()
    score, breakdown = grader.evaluate(env)
    # Clamp all sub-scores too
    for key in breakdown:
        if isinstance(breakdown[key], (int, float)) and key != 'explanation':
            breakdown[key] = clamp_score(breakdown[key])
    return {
        "score": clamp_score(score),
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
    agent = LLMAgent()
    done = False
    
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = local_env.step(action)
        
    grader = MultiFactorGrader()
    score, breakdown = grader.evaluate(local_env)
    local_env.logger.plot_history(filename=f"baseline_{difficulty}_log.png")
    # In production, cache/DB persistence would occur here
