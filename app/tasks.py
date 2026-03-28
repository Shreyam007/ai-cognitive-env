import random
import uuid
from typing import List
from app.models import TaskItem

class ScenarioGenerator:
    def __init__(self, difficulty: str = "easy"):
        self.difficulty = difficulty
        self.spawned_interruptions = 0
        
    def generate_initial(self, difficulty_modifier: float) -> List[TaskItem]:
        tasks = []
        if self.difficulty == "easy":
            tasks.append(self._make_task("Study Math", 2, 4.0, 8.0))
            tasks.append(self._make_task("Read Book", 1, 2.0, 10.0))
        elif self.difficulty == "medium":
            tasks.append(self._make_task("Project Proposal", 4, 3.0, 6.0))
            tasks.append(self._make_task("Reply Emails", 2, 1.0, 4.0))
            tasks.append(self._make_task("Fix Bug", 3, 2.0, 10.0))
        elif self.difficulty == "hard":
            tasks.append(self._make_task("Production Outage", 5, 4.0, 5.0))
            tasks.append(self._make_task("Client Meeting Prep", 4, 3.0, 8.0))
            tasks.append(self._make_task("Write Report", 3, 5.0, 12.0))
            tasks.append(self._make_task("Code Review", 2, 2.0, 15.0))
        return tasks
        
    def maybe_interrupt(self, current_time: float, difficulty_modifier: float) -> List[TaskItem]:
        new_tasks = []
        if self.difficulty == "easy":
            return new_tasks
            
        if self.difficulty == "medium" and self.spawned_interruptions < 1 and current_time > 4.0:
            if random.random() < 0.3 * difficulty_modifier:
                new_tasks.append(self._make_task("Urgent Request", 4, 1.0, current_time + 4.0))
                self.spawned_interruptions += 1
                
        if self.difficulty == "hard" and self.spawned_interruptions < 3:
            if random.random() < 0.4 * difficulty_modifier:
                new_tasks.append(self._make_task(f"Ad-hoc Crisis {self.spawned_interruptions+1}", 5, 2.0, current_time + 3.0))
                self.spawned_interruptions += 1
                
        return new_tasks

    def _make_task(self, title: str, priority: int, duration_required: float, deadline: float) -> TaskItem:
        diff_map = {"easy": 1, "medium": 3, "hard": 5}
        d_val = diff_map.get(self.difficulty, 1)
        return TaskItem(
            id=str(uuid.uuid4())[:8],
            title=title,
            priority=priority,
            difficulty=d_val,
            deadline=deadline,
            duration_required=duration_required
        )
