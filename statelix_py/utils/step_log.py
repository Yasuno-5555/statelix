"""Step logging utility for tracking analysis execution."""
from datetime import datetime


class StepLogger:
    """Simple logger for tracking analysis steps."""
    
    def __init__(self):
        self._steps = []
    
    def start_step(self, model: str, params: dict) -> str:
        """
        Start a new analysis step.
        
        Args:
            model: Name of the model being executed
            params: Parameters for the model
            
        Returns:
            Step ID for tracking
        """
        step_id = f"step_{len(self._steps):04d}"
        step_info = {
            "id": step_id,
            "model": model,
            "params": params,
            "start_time": datetime.now(),
            "end_time": None,
            "status": "running",
            "results": None
        }
        self._steps.append(step_info)
        print(f"[{step_id}] Start: {model} with {params}")
        return step_id
    
    def complete_step(self, step_id: str, status: str, results: dict = None):
        """
        Mark a step as completed.
        
        Args:
            step_id: The step ID returned from start_step
            status: Final status ('success' or 'error')
            results: Optional results dictionary
        """
        for step in self._steps:
            if step["id"] == step_id:
                step["end_time"] = datetime.now()
                step["status"] = status
                step["results"] = results
                elapsed = (step["end_time"] - step["start_time"]).total_seconds()
                print(f"[{step_id}] Complete: {status} ({elapsed:.3f}s)")
                return
        
        print(f"[WARNING] Step {step_id} not found")
    
    def get_history(self) -> list:
        """Return all logged steps."""
        return self._steps.copy()
    
    def clear(self):
        """Clear step history."""
        self._steps.clear()
