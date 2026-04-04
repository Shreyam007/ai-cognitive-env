import os
import json
from openai import OpenAI
from app.environment import CognitiveEnv
from app.agents import BaselineAgent
from app.tasks import ScenarioGenerator

# 1. Environment Configuration (Alignment with Pre-Submission Checklist)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    # 2. Local Environment Setup
    # Note: In the final evaluation, the platform might use its own environment runner,
    # but inference.py must demonstrate the standard interaction loop.
    env = CognitiveEnv()
    scenario = ScenarioGenerator("medium")
    
    # Reset Environment
    obs = env.reset(seed=42, scenario_generator=scenario)
    
    # 3. Mandatory Structured Logging: START
    print(f"START: {json.dumps(obs if isinstance(obs, dict) else obs.model_dump())}")
    
    agent = BaselineAgent()
    done = False
    step_count = 0
    max_steps = 50
    
    while not done and step_count < max_steps:
        # Agent Decision
        action = agent.decide(obs)
        
        # Environment Step
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # 4. Mandatory Structured Logging: STEP
        print(f"STEP: {json.dumps(obs if isinstance(obs, dict) else obs.model_dump())}")
        
    # 5. Mandatory Structured Logging: END
    print(f"END: {json.dumps(info)}")

if __name__ == "__main__":
    main()
