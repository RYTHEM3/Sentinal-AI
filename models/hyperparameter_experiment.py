import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


print("=" * 60)
print("       SENTINAL AI - HYPERPARAMETER EXPERIMENT")
print("=" * 60)


# Load features
X_train = joblib.load("dataset/processed/features/X_train.pkl")
X_val = joblib.load("dataset/processed/features/X_val.pkl")

y_train = joblib.load("dataset/processed/features/y_train.pkl")
y_val = joblib.load("dataset/processed/features/y_val.pkl")


# Experiment values
C_values = [0.5, 1.0, 2.0]

results = []


for c in C_values:

    print(f"\nTraining model with C = {c}")

    model = LogisticRegression(
        C=c,
        max_iter=1000,
        solver="lbfgs",
        class_weight=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    results.append({
        "C": c,
        "Accuracy": accuracy,
        "Macro F1": macro_f1
    })

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))


# Display comparison
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("EXPERIMENT COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

best_index = results_df["Macro F1"].idxmax()

print("\nBest C:", results_df.loc[best_index, "C"])
print("Best Macro F1:", results_df.loc[best_index, "Macro F1"])


# Save results
results_df.to_csv(
    "evaluation/hyperparameter_results.csv",
    index=False
)

print("\nResults saved successfully!")