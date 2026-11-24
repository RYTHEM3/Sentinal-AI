import sys
import joblib
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
FEATURE_DIR = BASE_DIR / "dataset" / "processed" / "features"
OUTPUT_DIR = BASE_DIR / "evaluation" / "test_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))


# --------------------------------------------------
# 2. REGISTER CUSTOM MODEL CLASSES
# --------------------------------------------------

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

main_module = sys.modules["__main__"]

for class_name, module_name in class_locations.items():
    module = importlib.import_module(module_name)
    setattr(main_module, class_name, getattr(module, class_name))

print("Model classes registered successfully.")


# --------------------------------------------------
# 3. LOAD TEST DATA
# --------------------------------------------------

print("\nLoading untouched test dataset...")

X_test = joblib.load(FEATURE_DIR / "X_test.pkl")
y_test = joblib.load(FEATURE_DIR / "y_test.pkl")

y_test = np.asarray(y_test).ravel()

print("Test samples:", X_test.shape[0])
print("Features:", X_test.shape[1])


# --------------------------------------------------
# 4. MODEL CONFIGURATION
# --------------------------------------------------

models = {
    "Batch Gradient Descent":
        "logistic_regression_scratch_best.pkl",

    "Mini-Batch Gradient Descent":
        "mini_batch_logistic_regression.pkl",

    "Stochastic Gradient Descent":
        "stochastic_logistic_regression.pkl",

    "Newton-CG":
        "newtons_method_model.pkl",

    "Golden Section Search":
        "golden_section_search.pkl",

    "Sklearn Logistic Regression":
        "logistic_regression.pkl"
}


# --------------------------------------------------
# 5. EVALUATE MODELS
# --------------------------------------------------

results = []
class_labels = ["Negative", "Neutral", "Positive"]

for model_name, filename in models.items():

    print("\n" + "=" * 65)
    print("Evaluating:", model_name)
    print("=" * 65)

    model_path = MODEL_DIR / filename
    model = joblib.load(model_path)

    # Predict using saved model
    predictions = model.predict(X_test)
    predictions = np.asarray(predictions).ravel()

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )

    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )

    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    training_time = getattr(model, "training_time", np.nan)

    # Store results
    results.append({
        "Algorithm": model_name,
        "Test Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Training Time": training_time
    })

    # Print metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:\n")

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=class_labels,
        zero_division=0
    )

    print(report)

    # Save classification report
    report_path = OUTPUT_DIR / (
        model_name.lower().replace(" ", "_").replace("-", "_")
        + "_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{model_name} - Test Confusion Matrix")
    plt.tight_layout()

    image_name = (
        model_name.lower().replace(" ", "_").replace("-", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(OUTPUT_DIR / image_name, dpi=300)
    plt.close()

    print("Confusion matrix saved.")


# --------------------------------------------------
# 6. SAVE COMPARISON RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

results_df.to_csv(
    OUTPUT_DIR / "final_test_comparison.csv",
    index=False
)

print("\n" + "=" * 75)
print("FINAL TEST SET COMPARISON")
print("=" * 75)

print(results_df.to_string(index=False))

# --------------------------------------------------
# 7. GENERATE MACRO F1 COMPARISON GRAPH
# --------------------------------------------------

plt.figure(figsize=(12, 7))

plt.bar(
    results_df["Algorithm"],
    results_df["Macro F1"]
)

plt.title("Algorithm Comparison - Test Macro F1")
plt.xlabel("Optimization Algorithm")
plt.ylabel("Macro F1 Score")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "test_macro_f1_comparison.png",
    dpi=300
)

plt.close()

print("\nEvaluation completed successfully!")
print("Results saved in:", OUTPUT_DIR)
