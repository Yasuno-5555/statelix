
import sys
import os

# Ensure we can import statelix modules
sys.path.append(os.getcwd())

class DummyModel:
    def __init__(self):
        self.coef_ = [0.5, -0.2, 0.0]
        self.intercept_ = 1.0
        self.feature_names_in_ = ['Education', 'Age', 'Unrelated']

    def get_params(self):
        return {}
        
    def predict(self, X):
        return [0] * len(X)

try:
    from statelix_pkg.inquiry.narrative import Storyteller
    
    # Mock adapter since we might not have full sklearn integration in this env
    # But Storyteller uses adapters. Let's see if it works with a duck-typed object
    # The LinearAdapter expects a model with coef_ or feature_importances_
    
    model = DummyModel()
    story = Storyteller(model, feature_names=['Education', 'Age', 'Unrelated'])
    
    print("--- Storyteller Output ---")
    print(story.explain())
    print("------------------------")
    
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback to direct path import if package structure is complex in this env
    sys.path.append(os.path.join(os.getcwd(), 'statelix_pkg'))
    try:
        from inquiry.narrative import Storyteller
        model = DummyModel()
        story = Storyteller(model, feature_names=['Education', 'Age', 'Unrelated'])
        print("--- Storyteller Output (Retry) ---")
        print(story.explain())
    except Exception as e2:
        print(f"Retry failed: {e2}")

except Exception as e:
    print(f"Execution Error: {e}")
