"""
Project Manager Module
Handles saving and loading project state.
"""
import json
import pickle
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

class ProjectManager:
    """Manages project state for Statelix sessions."""
    
    PROJECT_EXT = ".stlx"
    
    def __init__(self):
        self.project_path = None
        self.project_name = "Untitled Project"
        self.created_at = None
        self.modified_at = None
    
    def save_project(self, filepath: str, data_manager) -> bool:
        """
        Save current project state to file.
        
        Parameters:
            filepath: Path to save the project
            data_manager: DataManager instance with current state
        """
        try:
            state = {
                'version': '1.0',
                'name': self.project_name,
                'created_at': self.created_at or datetime.now().isoformat(),
                'modified_at': datetime.now().isoformat(),
                'data': {
                    'df': data_manager.df.to_dict() if data_manager.df is not None else None,
                    'filename': data_manager.filename,
                    'history': data_manager.get_history(),
                    'weight_col': data_manager.get_weight_column() if hasattr(data_manager, 'get_weight_column') else None,
                    'value_labels': getattr(data_manager, '_value_labels', {})
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            self.project_path = filepath
            self.modified_at = state['modified_at']
            return True
            
        except Exception as e:
            print(f"Save project error: {e}")
            return False
    
    def load_project(self, filepath: str, data_manager) -> bool:
        """
        Load project state from file.
        
        Parameters:
            filepath: Path to the project file
            data_manager: DataManager instance to restore state to
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore DataFrame
            if state['data']['df'] is not None:
                df = pd.DataFrame.from_dict(state['data']['df'])
                data_manager.set_data(df, state['data']['filename'])
            
            # Restore history
            if hasattr(data_manager, '_history'):
                data_manager._history = state['data'].get('history', [])
            
            # Restore weight column
            w_col = state['data'].get('weight_col')
            if w_col and hasattr(data_manager, 'set_weight_column'):
                data_manager.set_weight_column(w_col)
            
            # Restore value labels
            if hasattr(data_manager, '_value_labels'):
                data_manager._value_labels = state['data'].get('value_labels', {})
            
            self.project_path = filepath
            self.project_name = state.get('name', 'Loaded Project')
            self.created_at = state.get('created_at')
            self.modified_at = state.get('modified_at')
            
            return True
            
        except Exception as e:
            print(f"Load project error: {e}")
            return False
    
    @staticmethod
    def get_recent_projects(max_count: int = 5) -> list:
        """Get list of recently opened projects (placeholder)."""
        # In a full implementation, this would read from a config file
        return []
