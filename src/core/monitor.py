import json
import os
import time
from src.config import Config

class Monitor:
    """
    Monitors the Engine state and saves it to a JSON file for the Dashboard.
    """
    def __init__(self, filepath="src/data/monitor.json"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    def update(self, state_dict):
        """
        Update the monitor file with the latest state.
        state_dict: Dictionary containing relevant metrics.
        """
        try:
            # Add timestamp
            state_dict['timestamp'] = time.time()
            state_dict['last_update'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Atomic write (write to temp then rename)
            temp_path = self.filepath + ".tmp"
            
            # Helper to convert numpy types
            def default(o):
                if hasattr(o, 'item'): return o.item()
                if hasattr(o, '__iter__'): return list(o)
                return str(o)
                
            with open(temp_path, 'w') as f:
                json.dump(state_dict, f, indent=4, default=default)
                
            os.replace(temp_path, self.filepath)
            
        except Exception as e:
            print(f"MONITOR UPDATE FAILED: {e}")
