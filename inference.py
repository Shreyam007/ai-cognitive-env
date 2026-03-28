import os
from app.environment import CognitiveEnv
from app.agents import BaselineAgent
from app.tasks import ScenarioGenerator

def main():
    print("Starting OpenEnv Inference Check...")
    env = CognitiveEnv()
    scenario = ScenarioGenerator("easy")
    obs = env.reset(seed=42, scenario_generator=scenario)
    agent = BaselineAgent()
    
    done = False
    step_count = 0
    while not done and step_count < 20: # fail-safe
        action = agent.decide(obs)
        obs, reward, done, info = env.step(action)
        step_count += 1
        
    print(f"Inference complete in {step_count} steps.")
    print("Final State:", env.state())

if __name__ == "__main__":
    main()
