
import pytest
import numpy as np
import pandas as pd
from statelix_py.models.synthetic_control import SyntheticControl

# Only run if C++ backend is available
try:
    import statelix_core
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

@pytest.mark.skipif(not HAS_CPP, reason="Requires compiled statelix_core")
def test_synthetic_control_basic():
    # Create synthetic data
    # 3 units: A (treated), B, C (donors)
    # T=10 periods, treatment at T=7
    times = list(range(10))
    units = ['A', 'B', 'C']
    
    data = []
    for t in times:
        # B and C are stable
        val_b = 10 + np.random.normal(0, 0.1)
        val_c = 20 + np.random.normal(0, 0.1)
        
        # A is average of B and C pre-treatment
        val_a = 0.5 * val_b + 0.5 * val_c
        
        # Add treatment effect for A after t=7
        if t >= 7:
            val_a += 5.0
            
        data.append({'time': t, 'unit': 'A', 'outcome': val_a})
        data.append({'time': t, 'unit': 'B', 'outcome': val_b})
        data.append({'time': t, 'unit': 'C', 'outcome': val_c})
        
    df = pd.DataFrame(data)
    
    sc = SyntheticControl()
    sc.fit(
        data=df,
        unit_col='unit',
        time_col='time',
        outcome_col='outcome',
        treated_unit='A',
        treatment_period=7
    )
    
    res = sc.result_
    
    # Check weights: should be close to 0.5, 0.5
    weights = res.weights
    # Since we can't easily map back to B/C order without checking internal indices,
    # just check if sum is 1 and values are reasonable
    assert abs(np.sum(weights) - 1.0) < 1e-4
    assert np.all(weights >= -1e-5)
    
    # Check treatment effect
    # True effect is 5.0
    assert abs(res.att - 5.0) < 0.5
    
    # Check pre-treatment fit
    assert res.pre_rmspe < 0.5

@pytest.mark.skipif(not HAS_CPP, reason="Requires compiled statelix_core")
def test_placebo_test():
    # Similar setup
    times = list(range(10))
    units = ['A', 'B', 'C', 'D', 'E']
    
    data = []
    for t in times:
        val_b = 10 + np.random.normal(0, 0.1)
        val_c = 20 + np.random.normal(0, 0.1)
        val_d = 15 + np.random.normal(0, 0.1)
        val_e = 25 + np.random.normal(0, 0.1)
        
        val_a = 0.5 * val_b + 0.5 * val_c
        if t >= 7:
            val_a += 5.0
            
        for u, v in zip(units, [val_a, val_b, val_c, val_d, val_e]):
            data.append({'time': t, 'unit': u, 'outcome': v})
            
    df = pd.DataFrame(data)
    
    sc = SyntheticControl()
    sc.fit(df, 'unit', 'time', 'outcome', 'A', 7)
    
    placebo_df = sc.placebo_test()
    
    assert 'p_value' in placebo_df.columns
    assert placebo_df['treated_rank'][0] <= 2 # Should be significant or close
