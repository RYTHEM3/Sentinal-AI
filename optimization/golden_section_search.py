
import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_DIR = BASE_DIR / "dataset" / "processed" / "features"
MODEL_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------


def load_data(filename):
    path = FEATURE_DIR / filename

    return joblib.load(path)
    


X_train = load_data("X_train.pkl")
X_val = load_data("X_val.pkl")

y_train = np.asarray(
    load_data("y_train.pkl")
).ravel().astype(int)

y_val = np.asarray(
    load_data("y_val.pkl")
).ravel().astype(int)


# --------------------------------------------------
# GOLDEN SECTION SEARCH LOGISTIC REGRESSION
# --------------------------------------------------

class GoldenSectionLogisticRegression:

    def __init__(
        self,
        num_features,
        num_classes=3,
        reg_strength=0.00002,
        tolerance=1e-5
    ):

        self.num_features = num_features
        self.num_classes = num_classes

        self.reg_strength = reg_strength
        self.tolerance = tolerance

        self.weights = np.zeros(
            (num_features, num_classes)
        )

        self.bias = np.zeros(num_classes)

        self.loss_history = []
        self.step_history = []

    # --------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------

    def softmax(self, z):

        z = z - np.max(
            z,
            axis=1,
            keepdims=True
        )

        exp_z = np.exp(z)

        return exp_z / np.sum(
            exp_z,
            axis=1,
            keepdims=True
        )

    # --------------------------------------------------
    # CROSS-ENTROPY LOSS
    # --------------------------------------------------

    def compute_loss(
        self,
        X,
        y,
        weights=None,
        bias=None
    ):

        if weights is None:
            weights = self.weights

        if bias is None:
            bias = self.bias

        logits = X @ weights + bias

        probabilities = self.softmax(logits)

        correct_probabilities = probabilities[
            np.arange(len(y)),
            y
        ]

        correct_probabilities = np.clip(
            correct_probabilities,
            1e-12,
            1.0
        )

        sample_weights = self.class_weights[y]

        cross_entropy = -np.mean(
        sample_weights * np.log(correct_probabilities)
        )
            

        regularization = (
            0.5
            * self.reg_strength
            * np.sum(weights ** 2)
        )

        return cross_entropy + regularization

    # --------------------------------------------------
    # COMPUTE GRADIENT
    # --------------------------------------------------

    def compute_gradients(self, X, y):

        num_samples = X.shape[0]

        logits = X @ self.weights + self.bias

        probabilities = self.softmax(logits)

        y_onehot = np.eye(
            self.num_classes
        )[y]

        sample_weights = self.class_weights[y]

        dz = (
            probabilities - y_onehot
        ) * sample_weights[:, np.newaxis]

        dw = (
            X.T @ dz
        ) / num_samples

        dw = dw + (
            self.reg_strength * self.weights
        )

        db = np.mean(
            dz,
            axis=0
        )

        return dw, db

    # --------------------------------------------------
    # GOLDEN SECTION SEARCH
    # WITH ADAPTIVE BRACKETING
    # --------------------------------------------------

    def golden_section_search(
        self,
        X,
        y,
        direction_weights,
        direction_bias
    ):

        # Objective function along the negative
        # gradient direction.

        def objective(step):

            candidate_weights = (
                self.weights
                - step * direction_weights
            )

            candidate_bias = (
                self.bias
                - step * direction_bias
            )

            return self.compute_loss(
                X,
                y,
                candidate_weights,
                candidate_bias
            )

        # ------------------------------------------
        # STEP 1: ADAPTIVE BRACKETING
        # ------------------------------------------

        alpha_previous = 0.0
        loss_previous = objective(alpha_previous)

        alpha_current = 1.0
        loss_current = objective(alpha_current)

        # If loss increases immediately,
        # the minimum is already bracketed.

        if loss_current >= loss_previous:

            left = 0.0
            right = 1.0

        else:

            # Expand the search interval until
            # the loss starts increasing.

            bracket_found = False

            for _ in range(10):

                alpha_next = alpha_current * 2

                loss_next = objective(alpha_next)

                if loss_next >= loss_current:

                    left = alpha_previous
                    right = alpha_next

                    bracket_found = True
                    break

                alpha_previous = alpha_current
                loss_previous = loss_current

                alpha_current = alpha_next
                loss_current = loss_next

            if not bracket_found:

                print(
                    "Warning: Could not fully bracket "
                    "the minimum. Using the best "
                    "step found during expansion."
                )

                return alpha_current

        # ------------------------------------------
        # STEP 2: GOLDEN SECTION SEARCH
        # ------------------------------------------

        golden_ratio = (
            np.sqrt(5) - 1
        ) / 2

        x1 = right - golden_ratio * (
            right - left
        )

        x2 = left + golden_ratio * (
            right - left
        )

        f1 = objective(x1)
        f2 = objective(x2)

        while (right - left) > self.tolerance:

            if f1 < f2:

                right = x2

                x2 = x1
                f2 = f1

                x1 = right - golden_ratio * (
                    right - left
                )

                f1 = objective(x1)

            else:

                left = x1

                x1 = x2
                f1 = f2

                x2 = left + golden_ratio * (
                    right - left
                )

                f2 = objective(x2)

        best_step = (left + right) / 2

        return best_step

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    


    def fit(self, X, y, epochs=50):

        print("\nTraining Logistic Regression "
            "using Golden Section Search...\n")
        
        class_counts = np.bincount(y,minlength=self.num_classes)
        self.class_weights = np.sqrt(len(y) / (self.num_classes * class_counts))

        self.class_weights /= np.mean(self.class_weights)

        print("Class weights:", self.class_weights)

        for epoch in range(epochs):

            # Step 1: Calculate gradient
            dw, db = self.compute_gradients(X, y)

            # Step 2: Calculate gradient norm
            gradient_norm = np.sqrt(
                np.sum(dw ** 2) + np.sum(db ** 2)
            )

            # Step 3: Normalize gradient direction
            if gradient_norm > 1e-12:

                dw = dw / gradient_norm
                db = db / gradient_norm

            # Step 4: Find optimal step size
            step_size = self.golden_section_search(
                X,
                y,
                dw,
                db
            )

            # Step 5: Update parameters
            self.weights -= step_size * dw
            self.bias -= step_size * db

            # Step 6: Calculate updated loss
            loss = self.compute_loss(X, y)

            self.loss_history.append(loss)
            self.step_history.append(step_size)

            print(
                f"Epoch {epoch + 1}: "
                f"Loss = {loss:.6f}, "
                f"Step Size = {step_size:.6f}, "
                f"Gradient Norm = {gradient_norm:.6f}"
            )

        return self


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    def predict(self, X):

        logits = X @ self.weights + self.bias

        probabilities = self.softmax(logits)

        return np.argmax(
            probabilities,
            axis=1
        )


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

start_time = time.time()
def main():
        # LOAD DATA
    X_train = load_data("X_train.pkl")
    X_val = load_data("X_val.pkl")

    y_train = np.asarray(load_data("y_train.pkl")).ravel().astype(int)
    y_val = np.asarray(load_data("y_val.pkl")).ravel().astype(int)

    # START TIMER
    start_time = time.time()
    
    model = GoldenSectionLogisticRegression(
        num_features=X_train.shape[1],
        num_classes=3,
        reg_strength=0.00002,
        tolerance=1e-5
    )

    model.fit(
        X_train,
        y_train,
        epochs=200
    )

    training_time = time.time() - start_time


    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    train_predictions = model.predict(X_train)

    val_predictions = model.predict(X_val)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro"
    )

    final_loss = model.compute_loss(
        X_train,
        y_train
    )


    print("\n========== RESULTS ==========")

    print(f"Training Time: {training_time:.2f} seconds")

    print(f"Training Accuracy: {train_accuracy:.4f}")

    print(f"Validation Accuracy: {val_accuracy:.4f}")

    print(f"Macro F1 Score: {macro_f1:.4f}")

    print(f"Final Training Loss: {final_loss:.6f}")


    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_val,
            val_predictions,
            labels=[0, 1, 2],
            target_names=[
                "Negative",
                "Neutral",
                "Positive"
            ],
            zero_division=0
        )
    )


    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        y_val,
        val_predictions,
        labels=[0, 1, 2]
    )

    print("\nConfusion Matrix:\n")

    print(cm)


    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    model_path = (
        MODEL_DIR / "golden_section_search.pkl"
    )

    with open(model_path, "wb") as file:

        pickle.dump(model, file)

    print(f"\nModel saved to: {model_path}")


    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    results = pd.DataFrame([
        {
            "Algorithm": "Golden Section Search",
            "Training Accuracy": train_accuracy,
            "Validation Accuracy": val_accuracy,
            "Macro F1": macro_f1,
            "Final Loss": final_loss,
            "Training Time": training_time
        }
    ])

    results_path = (
        EVALUATION_DIR / "golden_section_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    print(f"Results saved to: {results_path}")


    # --------------------------------------------------
    # PLOT LOSS
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(model.loss_history) + 1),
        model.loss_history,
        marker="o"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Training Loss")

    plt.title("Golden Section Search - Loss Curve")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "golden_section_loss.png"
    )

    plt.show()


    # --------------------------------------------------
    # PLOT STEP SIZE
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(model.step_history) + 1),
        model.step_history,
        marker="o"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Selected Step Size")

    plt.title("Golden Section Search - Step Size")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "golden_section_step_size.png"
    )

    plt.show()


    print("\nGolden Section Search training completed!")
    
if __name__ == "__main__":
    main()
