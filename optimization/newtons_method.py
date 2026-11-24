
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import time

from scipy.sparse.linalg import LinearOperator, cg

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class NewtonLogisticRegression:

    def __init__(
        self,
        reg_strength=0.00002,
        damping=0.001,
        epochs=5,
        cg_iterations=10
    ):

        self.reg_strength = reg_strength
        self.damping = damping
        self.epochs = epochs
        self.cg_iterations = cg_iterations

        self.weights = None
        self.bias = None
        self.loss_history = []
        self.training_time = 0

    # Softmax
    def softmax(self, z):

        z = z - np.max(z, axis=1, keepdims=True)

        exp_z = np.exp(z)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Calculate loss
    def compute_loss(self, X, y):

        probabilities = self.softmax(
            X @ self.weights + self.bias
        )

        correct_probs = probabilities[
            np.arange(len(y)), y
        ]

        loss = -np.mean(
            np.log(correct_probs + 1e-15)
        )

        loss += (
            self.reg_strength / 2
        ) * np.sum(self.weights ** 2)

        return loss

    # Calculate gradient
    def compute_gradient(self, X, y, probabilities):

        n_samples = X.shape[0]

        dz = probabilities.copy()

        dz[np.arange(n_samples), y] -= 1

        dw = (
            X.T @ dz
        ) / n_samples

        dw += self.reg_strength * self.weights

        db = np.mean(dz, axis=0)

        return dw, db

    # Hessian-vector product
    def hessian_vector_product(
        self,
        X,
        probabilities,
        vector
    ):

        n_samples, n_features = X.shape

        n_classes = self.weights.shape[1]

        vector_weights = vector[:n_features * n_classes]
        vector_weights = vector_weights.reshape(
            n_features, n_classes
        )

        vector_bias = vector[n_features * n_classes:]

        # Perturbation in logits
        u = X @ vector_weights + vector_bias

        # Softmax Hessian-vector product
        weighted_mean = np.sum(
            probabilities * u,
            axis=1,
            keepdims=True
        )

        s = probabilities * (u - weighted_mean)

        hv_weights = (X.T @ s) / n_samples

        hv_weights += (
            self.reg_strength + self.damping
        ) * vector_weights

        hv_bias = np.mean(s, axis=0)

        hv_bias += self.damping * vector_bias

        return np.concatenate([
            np.asarray(hv_weights).reshape(-1),
            hv_bias
        ])

    # Training
    def fit(self, X, y):

        n_samples, n_features = X.shape

        n_classes = 3

        self.weights = np.zeros(
            (n_features, n_classes)
        )

        self.bias = np.zeros(n_classes)

        self.loss_history = []

        parameter_count = (
            n_features * n_classes + n_classes
        )

        print("\nTraining Newton-CG Logistic Regression...")

        start_time = time.time()

        for epoch in range(self.epochs):

            probabilities = self.softmax(
                X @ self.weights + self.bias
            )

            # Calculate gradient
            dw, db = self.compute_gradient(
                X, y, probabilities
            )

            gradient = np.concatenate([
                dw.reshape(-1),
                db
            ])

            # Hessian as a LinearOperator
            hessian = LinearOperator(
                (parameter_count, parameter_count),
                matvec=lambda v: self.hessian_vector_product(
                    X, probabilities, v
                ),
                dtype=np.float64
            )

            # Solve H * direction = gradient
            direction, info = cg(
                hessian,
                gradient,
                maxiter=self.cg_iterations,
                rtol=1e-4,
                atol=0.0
            )

            if info < 0:
                print("Warning: Conjugate Gradient failed.")
                break

            # Split Newton direction
            direction_weights = direction[
                :n_features * n_classes
            ].reshape(n_features, n_classes)

            direction_bias = direction[
                n_features * n_classes:
            ]

            # Backtracking line search
            old_loss = self.compute_loss(X, y)

            step_size = 1.0

            directional_derivative = np.dot(
                gradient, direction
            )

            old_weights = self.weights.copy()
            old_bias = self.bias.copy()

            while step_size > 1e-6:

                self.weights = (
                    old_weights -
                    step_size * direction_weights
                )

                self.bias = (
                    old_bias -
                    step_size * direction_bias
                )

                new_loss = self.compute_loss(X, y)

                if new_loss <= (
                    old_loss -
                    1e-4 * step_size *
                    directional_derivative
                ):
                    break

                step_size *= 0.5

            if step_size <= 1e-6:
                self.weights = old_weights
                self.bias = old_bias
                print("Line search stopped: no suitable step.")
                break

            self.loss_history.append(new_loss)

            print(
                f"Epoch {epoch + 1}: "
                f"Loss = {new_loss:.4f}, "
                f"Step Size = {step_size:.4f}, "
                f"CG Info = {info}"
            )

        self.training_time = time.time() - start_time

        print("\nTraining completed!")

        print(
            "Training Time:",
            round(self.training_time, 2),
            "seconds"
        )

    # Prediction
    def predict(self, X):

        probabilities = self.softmax(
            X @ self.weights + self.bias
        )

        return np.argmax(probabilities, axis=1)


if __name__ == "__main__":

    print("=" * 65)
    print(" SENTINAL AI - NEWTON-CG METHOD")
    print("=" * 65)

    # Load data
    X_train = joblib.load(
        "dataset/processed/features/X_train.pkl"
    )

    X_val = joblib.load(
        "dataset/processed/features/X_val.pkl"
    )

    y_train = np.asarray(
        joblib.load(
            "dataset/processed/features/y_train.pkl"
        ),
        dtype=int
    )

    y_val = np.asarray(
        joblib.load(
            "dataset/processed/features/y_val.pkl"
        ),
        dtype=int
    )

    # Initialize model
    model = NewtonLogisticRegression(
        reg_strength=0.00002,
        damping=0.001,
        epochs=5,
        cg_iterations=10
    )

    # Train
    model.fit(X_train, y_train)

    # Predictions
    train_predictions = model.predict(X_train)

    val_predictions = model.predict(X_val)

    # Evaluation
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
        average="macro",
        zero_division=0
    )

    final_loss = (
        model.loss_history[-1]
        if model.loss_history
        else model.compute_loss(X_train, y_train)
    )

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print("Training Accuracy:", round(train_accuracy, 4))
    print("Validation Accuracy:", round(val_accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Final Loss:", round(final_loss, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            val_predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_val,
            val_predictions
        )
    )

    # Save model
    joblib.dump(
        model,
        "models/newtons_method_model.pkl"
    )

    # Save results
    results = pd.DataFrame([{
        "Optimization": "Newton-CG",
        "Epochs": len(model.loss_history),
        "Training Accuracy": train_accuracy,
        "Validation Accuracy": val_accuracy,
        "Macro F1": macro_f1,
        "Final Loss": final_loss,
        "Training Time": model.training_time
    }])

    results.to_csv(
        "evaluation/newtons_method_results.csv",
        index=False
    )

    # Plot loss
    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, len(model.loss_history) + 1),
        model.loss_history,
        marker="o"
    )

    plt.title("Newton-CG Method - Loss Curve")
    plt.xlabel("Newton Iteration")
    plt.ylabel("Cross Entropy Loss")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "evaluation/newtons_method_loss.png"
    )

    plt.show()

    print("\nModel saved successfully!")
    print("Evaluation results saved!")
    print("Loss graph saved!")

    print("\nSENTINAL AI NEWTON-CG TRAINING COMPLETED!")
