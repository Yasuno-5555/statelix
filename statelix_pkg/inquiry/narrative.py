
from .adapters import BaseAdapter, LinearAdapter, BayesAdapter, CausalAdapter
from statelix.causal.core import BaseCausalModel

class Storyteller:
    def __init__(self, model, feature_names=None):
        if isinstance(model, BaseAdapter):
            self.adapter = model
        elif isinstance(model, BaseCausalModel):
            self.adapter = CausalAdapter(model)
        elif hasattr(model, 'aic') or hasattr(model, 'coef_'):
            self.adapter = LinearAdapter(model)
        elif hasattr(model, 'map_theta'):
            self.adapter = BayesAdapter(model)
        else:
            self.adapter = LinearAdapter(model)
            
        self.feature_names = feature_names

    def explain(self):
        """Generates a narrative summary of the model."""
        coefs = self.adapter.get_coefficients() # dict {idx: val}
        is_causal = isinstance(self.adapter, CausalAdapter)
        
        lines = []
        lines.append("### Model Narrative")
        
        # Metrics
        metrics = self.adapter.get_metrics()
        if 'r2' in metrics:
            lines.append(f"The model explains {metrics['r2']:.1%} of the variance in the target variable.")
            
        # Causal Specific Metrics
        if is_causal:
             if 'effect' in metrics:
                  lines.append(f"Estimated Causal Effect: {metrics['effect']:.4f}")
                  if 'std_error' in metrics:
                       lines.append(f"Standard Error: {metrics['std_error']:.4f}")
             lines.append("")
        else:
             lines.append("")
             
        lines.append("**Key Drivers:**")
        
        sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for idx, val in sorted_coefs:
            if abs(val) < 1e-4: continue
            
            name = f"Feature {idx}"
            if self.feature_names and idx < len(self.feature_names):
                name = self.feature_names[idx]
                
            direction = "positive" if val > 0 else "negative"
            
            # Phrase selection based on model type
            if is_causal:
                # Causal language
                phrase = f"- **{name}**: Has a {direction} **causal impact** (effect = {val:.4f})."
                interp = f"A 1-unit increase in {name} **causes** a {val:.4f} change in the outcome."
            else:
                # Association language
                phrase = f"- **{name}**: Has a {direction} association (beta = {val:.4f})."
                interp = f"A 1-unit increase in {name} corresponds to a {val:.4f} change in the outcome, holding others constant."
            
            lines.append(f"{phrase} {interp}")
        
        # Caveats / Assumptions for Causal Models
        if is_causal:
             assumptions = self.adapter.get_assumptions()
             if assumptions:
                  lines.append("")
                  lines.append("### Causal Assumptions & Caveats")
                  lines.append("> [!WARNING]")
                  lines.append("> Validity of these results depends on the following:")
                  for asm in assumptions:
                       lines.append(f"> - {asm}")

        return "\n".join(lines)
