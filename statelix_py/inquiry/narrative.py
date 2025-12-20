
from .adapters import BaseAdapter, LinearAdapter, BayesAdapter, CausalAdapter, DiscreteAdapter, SEMAdapter, BaseCausalModel
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
        """Generates a scaffolding summary of the model (FACTS + HINTS)."""
        coefs = self.adapter.get_coefficients()
        
        # Determine Narrative Style
        is_causal = isinstance(self.adapter, CausalAdapter)
        is_discrete = isinstance(self.adapter, DiscreteAdapter)
        is_sem = isinstance(self.adapter, SEMAdapter)
        
        lines = []
        lines.append("### 1. Analysis Facts (What the Data Says)")
        lines.append("_These are purely statistical results generated from your data._")
        lines.append("")
        
        # --- METRICS SECTION ---
        metrics = self.adapter.get_metrics()
        
        if is_sem:
            if 'proportion_mediated' in metrics:
                lines.append(f"- **Mediation Ratio:** {metrics['proportion_mediated']:.1%} of the effect passes through the mediator.")
                p_val = metrics.get('p_value_indirect', 1.0)
                sig = "statistically significant" if p_val < 0.05 else "not statistically significant"
                lines.append(f"- **Significance:** The indirect pathway is {sig} (p={p_val:.3f}).")
        elif is_discrete:
            if 'pseudo_r2' in metrics:
                lines.append(f"- **Model Fit:** Pseudo R-Squared is {metrics['pseudo_r2']:.3f}.")
        else:
            if 'r2' in metrics:
                lines.append(f"- **Explained Variance:** The model explains {metrics['r2']:.1%} of the variation in the target.")
            if 'effect' in metrics:
                lines.append(f"- **Estimated Parameter:** {metrics['effect']:.4f}")
        
        lines.append("")
        
        # --- COEFFICIENTS SECTION (FACTS) ---
        lines.append("**Observed Relationships:**")
        
        try:
            sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
        except:
            sorted_coefs = list(coefs.items())
        
        for name, val in sorted_coefs:
            if isinstance(val, (int, float)) and abs(val) < 1e-4: continue
            
            # Name Logic
            display_name = str(name)
            if isinstance(name, int) and self.feature_names and name < len(self.feature_names):
                display_name = self.feature_names[name]
            
            direction = "positive (+)" if isinstance(val, (int, float)) and val > 0 else "negative (-)"
            val_fmt = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)

            # --- SEM Facts ---
            if is_sem:
                if "->" in str(name):
                    lines.append(f"- **Path {display_name}**: Coefficient = {val_fmt}.")
                else:
                    lines.append(f"- **{name}**: Coefficient = {val_fmt}.")

            # --- Discrete Facts ---
            elif is_discrete:
                 if "/" in display_name and "x" not in display_name: continue
                 if self.feature_names: # Match names
                     for f in self.feature_names:
                         if f.lower() == display_name.lower():
                             display_name = f; break
                 lines.append(f"- **{display_name}**: {direction} association (coef = {val_fmt}).")

            # --- Linear/Causal Facts (Unified) ---
            else:
                # Intentionally avoiding "Causal Impact" wording even for CausalAdapter
                # We state the "Estimated Effect" which is a statistical fact given the model.
                lines.append(f"- **{display_name}**: {direction} association (coef = {val_fmt}).")

        lines.append("")
        lines.append("### 2. Interpretation Hints (Possibilities)")
        lines.append("_These are possible ways to interpret the facts. YOU must decide which are valid._")
        lines.append("")
        
        lines.append("> [!TIP]")
        lines.append("> **Positive Association (+):** As X increases, Y tends to increase.")
        lines.append("> **Negative Association (-):** As X increases, Y tends to decrease.")
        lines.append("> **Magnitude:** A coefficient of 0.5 means a 1-unit change in X is associated with a 0.5 change in Y.")
        
        if is_causal:
             lines.append("")
             lines.append("> [!WARNING]")
             lines.append("> **Causal Interpretation Risks:**")
             lines.append("> This model explicitly *attempts* to isolate a causal effect, but it relies on these assumptions:")
             assumptions = self.adapter.get_assumptions()
             if assumptions:
                 for asm in assumptions:
                     lines.append(f"> - {asm}")
             else:
                 lines.append("> - No unobserved confounders (Selection on Observables)")
             lines.append("> **If these assumptions fail, the result is just a correlation.**")
        
        lines.append("")
        lines.append("### 3. Conclusion")
        lines.append("*(This section is left blank for you. Based on the facts and hints above, what is your judgment?)*")
        
        return "\n".join(lines)
