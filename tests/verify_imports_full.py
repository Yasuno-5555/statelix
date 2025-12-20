import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("--- Verifying Statelix Imports ---")

try:
    print("1. Importing statelix.linear_model...")
    import statelix.linear_model as lm
    from statelix.linear_model import FitOLS
    print("   [OK] FitOLS found.")
except ImportError as e:
    print(f"   [FAIL] {e}")

try:
    print("2. Importing statelix.bayes...")
    import statelix.bayes as bayes
    try:
        from statelix.bayes import hmc_sample
        print("   [OK] hmc_sample found.")
    except ImportError:
        print("   [FAIL] hmc_sample NOT found in statelix.bayes")
        print(f"   Dir: {dir(bayes)}")
except ImportError as e:
    print(f"   [FAIL] {e}")

try:
    print("3. Importing statelix.graph...")
    import statelix.graph as graph
    print("   [OK] statelix.graph imported.")
except ImportError as e:
    print(f"   [FAIL] {e}")

try:
    print("4. Importing statelix.time_series...")
    import statelix.time_series as ts
    print("   [OK] statelix.time_series imported.")
except ImportError as e:
    print(f"   [FAIL] {e}")

try:
    print("5. Importing statelix.spatial...")
    import statelix.spatial as spatial
    print("   [OK] statelix.spatial imported.")
except ImportError as e:
    print(f"   [FAIL] {e}")

print("--- Verifying High-Level Models ---")
try:
    from statelix_py.models import StatelixOLS
    print("   [OK] StatelixOLS")
    from statelix_py.models import BayesianLogisticRegression
    print("   [OK] BayesianLogisticRegression")
    from statelix_py.models import StatelixGraph
    print("   [OK] StatelixGraph")
except ImportError as e:
    print(f"   [FAIL] High-level model import error: {e}")

print("--- Done ---")
