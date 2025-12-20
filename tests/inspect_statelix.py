import sys
import os
try:
    import statelix
    print("statelix imported")
    try:
        import statelix.causal
        print("statelix.causal imported")
        print(dir(statelix.causal))
    except ImportError as e:
        print(f"statelix.causal failed: {e}")
except ImportError as e:
    print(f"statelix failed: {e}")
