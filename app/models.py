from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class TaskItem(BaseModel):
    id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title or description")
    priority: int = Field(..., description="Priority level: 1 (Lowest) to 5 (Highest)")
    difficulty: int = Field(..., description="Task difficulty multiplier: 1 to 5")
    deadline: float = Field(..., description="Step index when this task is due")
    duration_required: float = Field(..., description="Total hours/steps needed to complete this task")
    duration_completed: float = Field(0.0, description="Hours/steps already spent on this task")
    status: Literal['pending', 'in_progress', 'completed', 'missed', 'deferred', 'rejected'] = Field(
        'pending', description="Current status of the task"
    )

class Observation(BaseModel):
    current_time_step: float = Field(..., description="Current time of day/step in the simulation (e.g. 0.0 to 16.0)")
    stress_level: float = Field(..., description="Current stress level (0.0 to 100.0). High stress penalizes performance.")
    energy_level: float = Field(..., description="Current energy level (0.0 to 100.0). Affects task efficiency.")
    work_streak: int = Field(..., description="Number of consecutive steps spent working without a break")
    fatigue_multiplier: float = Field(..., description="Multiplier representing efficiency loss from fatigue (>= 1.0)")
    active_tasks: List[TaskItem] = Field(..., description="Tasks currently available or in progress")
    completed_tasks: List[TaskItem] = Field(..., description="Tasks already completed")
    missed_tasks: List[TaskItem] = Field(..., description="Tasks missed due to passing deadline")

class Action(BaseModel):
    action_type: Literal[
        'schedule_task', 
        'prioritize_task', 
        'defer_task', 
        'split_task', 
        'reject_task', 
        'adjust_deadline', 
        'suggest_break'
    ] = Field(..., description="Type of action the agent wants to perform")
    
    task_id: Optional[str] = Field(None, description="Target task ID, required for task-specific actions")
    duration: Optional[float] = Field(None, description="Duration for 'schedule_task' or 'suggest_break' in steps/hours")
    chunks: Optional[int] = Field(None, description="Number of chunks to split into, for 'split_task'")
    new_deadline: Optional[float] = Field(None, description="Requested new deadline, for 'adjust_deadline'")
