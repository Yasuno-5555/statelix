
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaseCausalModel(ABC):
    """
    Abstract Base Class for Causal Inference Models.
    
    Attributes:
        effect_ (float or np.array): The estimated causal effect (e.g., coefficient of treatment).
        std_error_ (float or np.array): Standard error of the effect.
        ci_ (tuple or list): Confidence interval (lower, upper).
        p_value_ (float): P-value of the effect.
        assumptions (list[str]): List of assumptions required for validity (e.g., "Parallel Trends").
        params_ (pd.Series or dict): Full parameter estimates beyond the main effect.
    """
    
    def __init__(self):
        self.effect_ = None
        self.std_error_ = None
        self.ci_ = None
        self.p_value_ = None
        self.assumptions = []
        self.params_ = None
        self._is_fitted = False
        
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass
        
    def summary(self):
        """Returns a summary text or dataframe of the results."""
        if not self._is_fitted:
            return "Model not fitted."
            
        summary_str = f"Causal Effect Estimate: {self.effect_:.4f}\n"
        if self.std_error_:
            summary_str += f"Standard Error: {self.std_error_:.4f}\n"
        if self.p_value_:
            summary_str += f"P-value: {self.p_value_:.4f}\n"
        if self.ci_:
            summary_str += f"95% CI: [{self.ci_[0]:.4f}, {self.ci_[1]:.4f}]\n"
        
        if self.assumptions:
            summary_str += "\nAssumptions:\n"
            for asm in self.assumptions:
                summary_str += f"- {asm}\n"
                
        return summary_str
    
    def predict(self, X):
        """
        By default, causal models do NOT support general prediction.
        Method raises NotImplementedError to enforce distinction between
        Inference and Prediction.
        """
        raise NotImplementedError(
            "This Causal Model is for inference only (estimating effect sizes). "
            "It does not support general prediction."
        )
