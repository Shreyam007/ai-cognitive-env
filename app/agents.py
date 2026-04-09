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
        import os
        from openai import OpenAI
        
        # Initialize exactly as requested
        self.client = OpenAI(
            base_url=os.getenv("API_BASE_URL"),
            api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        )
        self.model = os.getenv("MODEL_NAME")
            
    def act(self, obs) -> Action:
        # Pre-processing observation
        if isinstance(obs, dict):
            obs = Observation(**obs)
        
        # Mandatory message structure: system: task planner, user: str(observation)
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
        return self.act(obs)
