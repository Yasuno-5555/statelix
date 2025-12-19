
from .adapters import BaseAdapter, LinearAdapter, BayesAdapter, CausalAdapter, DiscreteAdapter, SEMAdapter
from statelix.causal.core import BaseCausalModel
from statelix_py.models.discrete import OrderedModel, MultinomialLogit
from statelix_py.models.sem import PathAnalysis, MediationAnalysis

class Storyteller:
    def __init__(self, model, feature_names=None):
        if isinstance(model, BaseAdapter):
            self.adapter = model
        elif isinstance(model, BaseCausalModel):
            self.adapter = CausalAdapter(model)
        elif isinstance(model, (OrderedModel, MultinomialLogit)):
             self.adapter = DiscreteAdapter(model)
        elif isinstance(model, (PathAnalysis, MediationAnalysis)):
             self.adapter = SEMAdapter(model)
        elif hasattr(model, 'aic') or hasattr(model, 'coef_'):
            self.adapter = LinearAdapter(model)
        elif hasattr(model, 'map_theta'):
            self.adapter = BayesAdapter(model)
        else:
            self.adapter = LinearAdapter(model)
            
        self.feature_names = feature_names

    def explain(self):
        """Generates a narrative summary of the model."""
        coefs = self.adapter.get_coefficients()
        
        # Determine Narrative Style
        is_causal = isinstance(self.adapter, CausalAdapter)
        is_discrete = isinstance(self.adapter, DiscreteAdapter)
        is_sem = isinstance(self.adapter, SEMAdapter)
        
        lines = []
        lines.append("### Analysis Narrative")
        
        # --- METRICS SECTION ---
        metrics = self.adapter.get_metrics()
        
        if is_sem:
            if 'proportion_mediated' in metrics:
                lines.append(f"**Mediation Analysis Result:**")
                lines.append(f"The analysis reveals that {metrics['proportion_mediated']:.1%} of the total effect is mediated (indirect).")
                p_val = metrics.get('p_value_indirect', 1.0)
                sig = "statistically significant" if p_val < 0.05 else "not significant"
                lines.append(f"The indirect effect is **{sig}** (p={p_val:.3f}).")
                lines.append("")
        elif is_discrete:
            if 'pseudo_r2' in metrics:
                lines.append(f"Model Fit: Pseudo R-Squared is {metrics['pseudo_r2']:.3f}.")
            lines.append("")
        else:
            if 'r2' in metrics:
                lines.append(f"The model explains {metrics['r2']:.1%} of the variance in the target variable.")
            if 'effect' in metrics:
                  lines.append(f"Estimated Causal Effect: {metrics['effect']:.4f}")
            lines.append("")

        # --- COEFFICIENTS SECTION ---
        lines.append("**Key Drivers & Relationships:**")
        
        try:
            sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
        except:
            sorted_coefs = list(coefs.items())
        
        for name, val in sorted_coefs:
            if isinstance(val, (int, float)) and abs(val) < 1e-4: continue
            
            # Name Resolution Logic
            display_name = str(name)
            
            # Case 1: Keys are indices (e.g. Scikit-learn/Linear)
            if isinstance(name, int) and self.feature_names and name < len(self.feature_names):
                display_name = self.feature_names[name]
            
            # Case 2: Keys are already names (e.g. Statsmodels/Discrete)
            # If feature_names provided, we might want to map case-insensitively or just trust the model key.
            # Usually statsmodels returns column names directly.
            pass  

            direction = "positive" if isinstance(val, (int, float)) and val > 0 else "negative"
            val_fmt = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)

            # --- SEM Narrative ---
            if is_sem:
                if "->" in str(name):
                    lines.append(f"- **Path {display_name}**: Coefficient is {val_fmt}.")
                else:
                    if name == "Indirect Effect":
                         lines.append(f"- **Indirect Effect** (Mechanism): {val_fmt}. This represents the pathway through the mediator.")
                    elif name == "Direct Effect":
                         lines.append(f"- **Direct Effect**: {val_fmt}. The effect remaining after accounting for the mediator.")
                    elif name == "Total Effect":
                         lines.append(f"- **Total Effect** (Overall Impact): {val_fmt}. The overall influence of X on Y.")

            # --- Discrete Narrative ---
            elif is_discrete:
                 # Check if this is a cut point
                 if "/" in display_name and "x" not in display_name: # Simple heuristic for cut-points "0/1"
                      continue # Skip display of cut points in main driver list

                 # Match names from feature_names just in case user provided them
                 if self.feature_names:
                     for f in self.feature_names:
                         if f.lower() == display_name.lower():
                             display_name = f
                             break
                             
                 lines.append(f"- **{display_name}**: {direction.capitalize()} influence (coef = {val_fmt}).")
                 if direction == "positive":
                     lines.append(f"  Higher values of this feature increase the likelihood of higher categories.")
                 else:
                     lines.append(f"  Higher values decrease the likelihood of higher categories.")

            # --- Causal Narrative ---
            elif is_causal:
                lines.append(f"- **{display_name}**: Has a {direction} **causal impact** (effect = {val_fmt}).")
                lines.append(f"  A 1-unit increase causes a {val_fmt} change in the outcome.")

            # --- Standard Linear Narrative ---
            else:
                lines.append(f"- **{display_name}**: Has a {direction} association (beta = {val_fmt}).")

        # --- CAVEATS ---
        if is_causal:
             assumptions = self.adapter.get_assumptions()
             if assumptions:
                  lines.append("")
                  lines.append("### Causal Assumptions")
                  lines.append("> [!WARNING]")
                  for asm in assumptions:
                       lines.append(f"> - {asm}")

        return "\n".join(lines)
