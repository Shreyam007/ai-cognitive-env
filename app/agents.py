import random
import os
import json
from openai import OpenAI
from app.models import Action, Observation

class BaseAgent:
    def decide(self, obs: Observation) -> Action:
        raise NotImplementedError

class RandomAgent(BaseAgent):
    def decide(self, obs) -> Action:
        if isinstance(obs, dict): obs = Observation(**obs)
        actions = ['schedule_task', 'suggest_break', 'prioritize_task']
        op = random.choice(actions)
        tid = random.choice(obs.active_tasks).id if obs.active_tasks else None
        
        return Action(action_type=op, task_id=tid, duration=1.0)
        
class RuleBasedAgent(BaseAgent):
    def decide(self, obs) -> Action:
        if isinstance(obs, dict): obs = Observation(**obs)
        if obs.stress_level > 85 or obs.energy_level < 20:
            return Action(action_type='suggest_break', duration=1.5)
            
        if obs.active_tasks:
            sorted_tasks = sorted(obs.active_tasks, key=lambda x: x.deadline)
            return Action(action_type='schedule_task', task_id=sorted_tasks[0].id, duration=0.5)
            
        return Action(action_type='suggest_break', duration=1.0)

class LLMAgent(BaseAgent):
    def __init__(self):
        # MANDATORY: Official Platform Credentials (LiteLLM Proxy)
        # Using exact snippet from requirement: prioritize API_KEY then HF_TOKEN
        import os
        from openai import OpenAI
        
        # Pull from environment. Using .get() for the flexible key but failing if neither exists.
        base_url = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
        model_name = os.environ.get("MODEL_NAME")
        
        if not base_url or not api_key:
            raise KeyError(f"CRITICAL: Missing LLM configuration. API_BASE_URL: {'found' if base_url else 'MISSING'}, API_KEY/HF_TOKEN: {'found' if api_key else 'MISSING'}")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model_name
            
    def act(self, obs) -> Action:
        # Pre-processing observation
        if isinstance(obs, dict):
            obs = Observation(**obs)
        
        # NO FALLBACK: This method MUST trigger an API call.
        print("LLM CALL USING PROXY", flush=True)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a task planner."},
                {"role": "user", "content": str(obs)}
            ],
            max_tokens=150
        )
        
        content = response.choices[0].message.content.strip()
        # Handle potential markdown from LLM
        if content.startswith("```json"): content = content[7:]
        elif content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]
        content = content.strip()
        
        action_data = json.loads(content)
        return Action(**action_data)

    def decide(self, obs) -> Action:
        # Legacy support for main.py while transitioning
        return self.act(obs)
