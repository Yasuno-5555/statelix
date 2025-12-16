import sys
import os

print("--- Statelix Release Verification ---")
try:
    import statelix
    if hasattr(statelix, '__version__'):
        print(f"Statelix Version: {statelix.__version__}")
    else:
        print("Statelix Version: (Namespace Package / No __init__ detected)")
except Exception as e:
    print(f"Error importing statelix: {e}")
    sys.exit(1)

try:
    import statelix.psm as psm
    print(f"Module Loaded: {psm.__doc__}")
except ImportError as e:
    print(f"FAILED to import statelix.psm: {e}")
    sys.exit(1)

try:
    import statelix.panel as panel
    print(f"Module Loaded: {panel.__doc__}")
except ImportError as e:
    print(f"FAILED to import statelix.panel: {e}")
    sys.exit(1)

try:
    import statelix.hmc as hmc
    print(f"Module Loaded: {hmc.__doc__}")
except ImportError as e:
    print(f"FAILED to import statelix.hmc: {e}")
    sys.exit(1)

print("\nSUCCESS: All modules (psm, panel, hmc) loaded and package structure is valid.")
