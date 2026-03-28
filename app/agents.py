import random
import os
import json
from openai import OpenAI
from app.models import Action, Observation

class BaseAgent:
    def decide(self, obs: Observation) -> Action:
        raise NotImplementedError

class RandomAgent(BaseAgent):
    def decide(self, obs: Observation) -> Action:
        actions = ['schedule_task', 'suggest_break', 'prioritize_task']
        op = random.choice(actions)
        tid = random.choice(obs.active_tasks).id if obs.active_tasks else None
        
        return Action(action_type=op, task_id=tid, duration=1.0)
        
class RuleBasedAgent(BaseAgent):
    def decide(self, obs: Observation) -> Action:
        if obs.stress_level > 85 or obs.energy_level < 20:
            return Action(action_type='suggest_break', duration=1.5)
            
        if obs.active_tasks:
            sorted_tasks = sorted(obs.active_tasks, key=lambda x: x.deadline)
            return Action(action_type='schedule_task', task_id=sorted_tasks[0].id, duration=0.5)
            
        return Action(action_type='suggest_break', duration=1.0)

class BaselineAgent(BaseAgent):
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.groq_api_key:
            self.client = OpenAI(api_key=self.groq_api_key, base_url="https://api.groq.com/openai/v1")
            self.model = "llama-3.1-8b-instant"
        elif self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
            self.model = "gpt-4o-mini"
        else:
            self.client = None
            self.model = None
            
    def decide(self, obs: Observation) -> Action:
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
                max_tokens=150,
                response_format={ "type": "json_object" }
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
            print(f"Fallback due to LLM parsing issue: {e}")
            return RuleBasedAgent().decide(obs)
