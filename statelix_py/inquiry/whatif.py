
import numpy as np
import pandas as pd
from .adapters import BaseAdapter, LinearAdapter

class WhatIf:
    def __init__(self, model, feature_names=None):
        """
        Counterfactual Engine.
        
        Args:
            model: Statelix Result object or Adapter.
            feature_names: List of feature names corresponding to X columns.
        """
        if isinstance(model, BaseAdapter):
            self.adapter = model
        elif hasattr(model, 'aic') or hasattr(model, 'coef_'):
            self.adapter = LinearAdapter(model)
        else:
            self.adapter = LinearAdapter(model)
            
        self.feature_names = feature_names

    def simulate(self, base_X: np.ndarray, changes: dict):
        """
        Simulate a change.
        
        Args:
            base_X: 1D or 2D array representing baseline state.
            changes: Dict of {feature_index_or_name: value_or_function}
                     e.g. {0: 1.5} (Set col 0 to 1.5)
                     e.g. {'tax': lambda x: x + 0.05} (Increase tax by 0.05)
                     
        Returns:
            Dict with 'baseline', 'scenario', 'delta', 'pct_change'
        """
        base_X = np.atleast_2d(base_X)
        scenario_X = base_X.copy()
        
        # Apply changes
        for key, modification in changes.items():
            # Resolve index
            idx = -1
            if isinstance(key, int):
                idx = key
            elif isinstance(key, str) and self.feature_names:
                try:
                    idx = self.feature_names.index(key)
                except ValueError:
                    continue
            
            if idx == -1 or idx >= base_X.shape[1]:
                continue
                
            # Apply modification
            if callable(modification):
                scenario_X[:, idx] = modification(scenario_X[:, idx])
            else:
                # Additive or Absolute?
                # User should specify absolute value or use lambda.
                # Here we assume absolute assignment if float
                scenario_X[:, idx] = modification

        # Predict
        y_base = self.adapter.predict(base_X)
        y_scen = self.adapter.predict(scenario_X)
        
        delta = y_scen - y_base
        pct = (delta / y_base) * 100.0 if not np.any(np.isclose(y_base, 0)) else np.zeros_like(delta)
        
        return {
            'baseline': y_base,
            'scenario': y_scen,
            'delta': delta,
            'pct_change': pct
        }
