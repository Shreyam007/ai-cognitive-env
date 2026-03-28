import os
from app.environment import CognitiveEnv
from app.tasks import ScenarioGenerator
from app.grader import MultiFactorGrader
from app.agents import RandomAgent, RuleBasedAgent, BaselineAgent

def run_benchmark(difficulty="easy", seed=42):
    print(f"\n{'='*55}\n   SCENARIO: {difficulty.upper()} | SEED: {seed}\n{'='*55}")
    
    agents = {
        "Random": RandomAgent(),
        "RuleBased": RuleBasedAgent(),
        "LLMBaseline": BaselineAgent()
    }
    
    results = []
    
    for name, agent in agents.items():
        env = CognitiveEnv()
        scenario = ScenarioGenerator(difficulty)
        obs = env.reset(seed=seed, scenario_generator=scenario)
        
        done = False
        while not done:
            action = agent.decide(obs)
            obs, reward, done, info = env.step(action)
            
        grader = MultiFactorGrader()
        score, sub = grader.evaluate(env)
        
        env.logger.plot_history(filename=f"benchmark_{name.strip()}_{difficulty}.png")
        
        results.append({
            "agent": name,
            "score": score,
            "productivity": sub['completion_ratio'],
            "efficiency": sub['efficiency'],
            "insights": sub['explanation']
        })
        
        print(f"[{name.strip()}] Score: {score:.2f}")
        print(f"    Productivity Ratio: {sub['completion_ratio']:.2f} | Action Efficiency: {sub['efficiency']:.2f}")
        print(f"    Diagnostic: {sub['explanation']}\n")

    print(f"--- {difficulty.upper()} SCENARIO RANKINGS ---")
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, res in enumerate(results, 1):
        print(f" {i}. {res['agent']:<12} | Score: {res['score']:.2f}")
        
    return results

if __name__ == "__main__":
    if "GROQ_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ:
        print("\n[SECURITY WARNING] No GROQ_API_KEY or OPENAI_API_KEY found via env variables.")
        print("The LLMBaseline agent will safely run in offline-heuristic mode. For true LLM reasoning, export the key in your terminal.")
        
    run_benchmark("easy", seed=42)
    run_benchmark("hard", seed=42)
