
"""
Statelix: The Explanatory Intelligence - Demo
---------------------------------------------
This script demonstrates the "Inquiry-First" workflow of Statelix.
We simulate a policy intervention and ask Statelix to:
1. Infer the causal effect (Difference-in-Differences).
2. Explain the result in plain English (Narrative).
3. Simulate a counterfactual scenario (WhatIf).
"""

import numpy as np
import pandas as pd
from statelix.causal import DiffInDiff
from statelix.inquiry import Storyteller, WhatIf

def main():
    print("=== Statelix Inquiry Demo ===\n")

    # 1. Simulate Data (Policy Intervention)
    np.random.seed(42)
    n = 1000
    
    # 50% Treated, 50% Control
    Group = np.random.randint(0, 2, n)
    # 50% Pre-Period, 50% Post-Period
    Time = np.random.randint(0, 2, n)
    
    # True Causal Effect (ATT) = 5.0
    # Y = Base + 2*Group + 3*Time + 5*(Group*Time) + Noise
    Interaction = Group * Time
    Y = 10.0 + 2.0*Group + 3.0*Time + 5.0*Interaction + np.random.normal(0, 1.0, n)
    
    # Create DataFrame for clarity (optional, Model accepts arrays)
    df = pd.DataFrame({'Y': Y, 'Group': Group, 'Time': Time})
    
    print(f"Data Simulated: N={n}, True ATT=5.0")
    
    # 2. Fit Causal Model
    print("\n--- 1. Causal Inference (DiD) ---")
    did = DiffInDiff()
    did.fit(df['Y'], Group=df['Group'], Time=df['Time'])
    
    print(did.summary())
    
    # 3. Narrative Explanation
    print("\n--- 2. Inquiry: Narrative ---")
    # We label features for the storyteller
    story = Storyteller(did, feature_names=["PolicyEffect", "TreatmentGroup", "PostPeriod", "Intercept"])
    print(story.explain())
    
    # 4. Counterfactual Simulation
    print("\n--- 3. Inquiry: What If? ---")
    print("Scenario: What if NO policy was implemented? (Force interaction to 0)")
    
    wi = WhatIf(did)
    
    # For DiD, "Policy Effect" corresponds to the Interaction term (index 0).
    # We simulate a world where the Interaction is 0 for everyone (No Effect).
    # Note: WhatIf usually modifies input features X. 
    # Our DiD model's predict takes [Interaction, Group, Time, Intercept].
    # We construct a base input vector representing a treated unit in post period.
    # [1, 1, 1, 1] -> Interaction=1, Group=1, Time=1, Intercept=1
    
    base_unit = np.array([[1.0, 1.0, 1.0, 1.0]]) 
    
    # Scenario: Interaction becomes 0
    # But usually WhatIf modifies underlying features.
    # Here we simplify by modifying the design matrix directly for demonstration.
    
    scenario_res = wi.simulate(base_unit.flatten(), {0: lambda x: 0.0})
    
    print(f"Baseline Outcome (Policy Applied):     {scenario_res['baseline'][0]:.4f}")
    print(f"Counterfactual Outcome (No Policy):    {scenario_res['scenario'][0]:.4f}")
    print(f"Attributed Impact (Delta):             {scenario_res['delta'][0]:.4f}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
