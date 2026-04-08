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

class BaselineAgent(BaseAgent):
    def __init__(self):
        # Priority 1: Official Platform Credentials (LiteLLM Proxy)
        self.api_key = os.getenv("API_KEY")
        self.api_base_url = os.getenv("API_BASE_URL")
        self.model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        # Priority 2: Secondary Platform Fallback (HF_TOKEN)
        if not self.api_key:
            self.api_key = os.getenv("HF_TOKEN")
            
        if self.api_key:
            # Always use API_BASE_URL if provided, else default to OpenAI standard
            base_url = self.api_base_url if self.api_base_url else "https://api.openai.com/v1"
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            # Fallback for local development
            local_key = os.getenv("OPENAI_API_KEY")
            if local_key:
                self.client = OpenAI(api_key=local_key)
            else:
                self.client = None
            
    def decide(self, obs) -> Action:
        if isinstance(obs, dict):
            obs_dict = obs
            obs = Observation(**obs)
        else:
            obs_dict = obs.model_dump()
            
        if not self.client:
            if obs.active_tasks:
                sorted_tasks = sorted(obs.active_tasks, key=lambda x: x.deadline)
                closest = sorted_tasks[0]
                if closest.deadline - obs.current_time_step <= 2.0 and obs.energy_level > 10:
                    return Action(action_type='schedule_task', task_id=closest.id, duration=1.0)
                
            if obs.stress_level > 70 or obs.energy_level < 30:
                return Action(action_type='suggest_break', duration=1.0)
                
            if obs.active_tasks:
                sorted_tasks = sorted(obs.active_tasks, key=lambda x: (x.deadline, x.duration_required))
                return Action(action_type='schedule_task', task_id=sorted_tasks[0].id, duration=1.0)
            return Action(action_type='suggest_break', duration=1.0)
            
        prompt = f"""You are an advanced cognitive scheduling agent. 
Current State: Energy {obs.energy_level:.1f}, Stress {obs.stress_level:.1f}. Active Tasks: {len(obs.active_tasks)}.
Tasks: {[t.title + ' (Due: ' + str(t.deadline) + ')' for t in obs.active_tasks]}
Choose an action formatted as JSON matching this Action schema: {Action.model_json_schema()}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=150
            )
            content = response.choices[0].message.content
            
            # Strip potential markdown formatting that Llama outputs natively
            content = content.strip()
            if content.startswith("```json"): content = content[7:]
            elif content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
            content = content.strip()
            
            action_data = json.loads(content)
            # Default missing fields logically if LLM misses them
            if 'action_type' not in action_data:
                action_data['action_type'] = 'suggest_break'
            if 'duration' not in action_data: action_data['duration'] = 1.0
            
            return Action(**action_data)
        except Exception as e:
            # Redirect to stderr to ensure it shows in platform logs without polluting mandated stdout format
            import sys
            print(f"Fallback due to API/Parsing issue: {e}", file=sys.stderr)
            return RuleBasedAgent().decide(obs)
