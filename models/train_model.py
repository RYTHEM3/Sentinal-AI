import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

print("=" * 60)
print("          SENTINAL AI - MODEL TRAINING")
print("=" * 60)


# Load feature matrices
print("\nLoading TF-IDF features...")

X_train = joblib.load("dataset/processed/features/X_train.pkl")
X_val = joblib.load("dataset/processed/features/X_val.pkl")

y_train = joblib.load("dataset/processed/features/y_train.pkl")
y_val = joblib.load("dataset/processed/features/y_val.pkl")


# Initialize Logistic Regression
print("\nInitializing Logistic Regression...")

model = LogisticRegression(
    C=2.0,
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42
)


# Train model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# Validation predictions
print("\nGenerating validation predictions...")

y_pred = model.predict(X_val)


# Evaluation
accuracy = accuracy_score(y_val, y_pred)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro"
)

print("\n" + "=" * 60)
print("              VALIDATION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=["Negative", "Neutral", "Positive"]
    )
)


# Confusion matrix
cm = confusion_matrix(y_val, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Save model
joblib.dump(model, "models/logistic_regression.pkl")

print("\nModel saved to models/logistic_regression.pkl")

print("\nSENTINAL AI BASELINE MODEL COMPLETED!")