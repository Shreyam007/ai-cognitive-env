from app.environment import CognitiveEnv
from app.tasks import ScenarioGenerator
from app.grader import MultiFactorGrader
from app.models import Action

class TestAgent1: # Dumb rule based
    def decide(self, obs):
        if obs.stress_level > 90 or obs.energy_level < 15:
            return Action(action_type='suggest_break', duration=1.0)
        if obs.active_tasks:
            sorted_tasks = sorted(obs.active_tasks, key=lambda x: x.deadline)
            return Action(action_type='schedule_task', task_id=sorted_tasks[0].id, duration=1.0)
        return Action(action_type='suggest_break', duration=1.0)

class TestAgent2:
    def decide(self, obs):
        if obs.active_tasks:
            sorted_tasks = sorted(obs.active_tasks, key=lambda x: x.deadline)
            closest = sorted_tasks[0]
            # Push through if very urgent
            if closest.deadline - obs.current_time_step <= 2.0 and obs.energy_level > 10:
                return Action(action_type='schedule_task', task_id=closest.id, duration=1.0)
            
        if obs.stress_level > 70 or obs.energy_level < 30:
            return Action(action_type='suggest_break', duration=1.0)
            
        if obs.active_tasks:
            sorted_tasks = sorted(obs.active_tasks, key=lambda x: (x.deadline, x.duration_required))
            return Action(action_type='schedule_task', task_id=sorted_tasks[0].id, duration=1.0)
        return Action(action_type='suggest_break', duration=1.0)

class RandomA:
    import random
    def decide(self, obs):
        import random
        actions = ['schedule_task', 'suggest_break', 'prioritize_task']
        op = random.choice(actions)
        tid = random.choice(obs.active_tasks).id if obs.active_tasks else None
        return Action(action_type=op, task_id=tid, duration=1.0)

def evaluate(agent, diff):
    env = CognitiveEnv()
    scen = ScenarioGenerator(diff)
    obs = env.reset(seed=42, scenario_generator=scen)
    done = False
    while not done:
        action = agent.decide(obs)
        obs, r, done, info = env.step(action)
    grader = MultiFactorGrader()
    return grader.evaluate(env)[0]

print("EASY:")
print("Random: ", evaluate(RandomA(), 'easy'))
print("RuleBased: ", evaluate(TestAgent1(), 'easy'))
print("Smart: ", evaluate(TestAgent2(), 'easy'))

print("HARD:")
print("Random: ", evaluate(RandomA(), 'hard'))
print("RuleBased: ", evaluate(TestAgent1(), 'hard'))
print("Smart: ", evaluate(TestAgent2(), 'hard'))
