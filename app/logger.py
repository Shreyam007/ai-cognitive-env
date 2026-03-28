import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict

class SessionLogger:
    def __init__(self):
        self.history = []
        
    def log_step(self, time_step: float, stress: float, energy: float, action_type: str, action_target: str, reward: float):
        self.history.append({
            'time': time_step,
            'stress': stress,
            'energy': energy,
            'action': action_type,
            'target': action_target,
            'reward': reward
        })

    def plot_history(self, filename: str = "session_log.png"):
        if not self.history:
            return
            
        times = [entry['time'] for entry in self.history]
        stresses = [entry['stress'] for entry in self.history]
        energies = [entry['energy'] for entry in self.history]
        
        plt.figure(figsize=(10, 5))
        plt.plot(times, stresses, label='Stress', color='red')
        plt.plot(times, energies, label='Energy', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Level (0-100)')
        plt.title('Cognitive Load Environment: Stress & Energy over Time')
        plt.axhline(80, color='darkred', linestyle='--', alpha=0.5, label='Burnout Threshold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
