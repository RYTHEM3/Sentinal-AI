import sys
import joblib
import importlib
from pathlib import Path

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Original model class locations
class_locations = {
    "GoldenSectionLogisticRegression":
        "optimization.golden_section_search",

    "LogisticRegressionScratch":
        "optimization.logistic_regression_scratch",

    "MiniBatchLogisticRegression":
        "optimization.mini_batch_gradient_descent",

    "NewtonLogisticRegression":
        "optimization.newtons_method",

    "StochasticLogisticRegression":
        "optimization.stochastic_gradient_descent"
}

# Register original classes under __main__
main_module = sys.modules["__main__"]

for class_name, module_name in class_locations.items():
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    setattr(main_module, class_name, model_class)

print("Original model classes registered successfully.")

model_files = [
    "golden_section_search.pkl",
    "mini_batch_logistic_regression.pkl",
    "stochastic_logistic_regression.pkl",
    "newtons_method_model.pkl",
    "logistic_regression_scratch_best.pkl",
    "logistic_regression.pkl"
]

for filename in model_files:

    path = BASE_DIR / "models" / filename

    print("\n" + "=" * 55)
    print("FILE:", filename)

    try:
        model = joblib.load(path)

        print("Type:", type(model))
        print("Attributes:", list(vars(model).keys()))
        print("Prediction method:", hasattr(model, "predict"))

    except Exception as error:
        print("ERROR:", error)
