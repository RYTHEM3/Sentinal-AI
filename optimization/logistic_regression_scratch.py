
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class LogisticRegressionScratch:

    def __init__(self, learning_rate=5.0, epochs=1000, reg_strength=0.00002):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_strength = reg_strength

        self.weights = None
        self.bias = None
        self.loss_history = []

    # Softmax function
    def softmax(self, z):

        z = z - np.max(z, axis=1, keepdims=True)

        exp_z = np.exp(z)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Cross entropy loss
    def compute_loss(self, y_true, probabilities):

        n_samples = len(y_true)

        correct_probs = probabilities[
            np.arange(n_samples),
            y_true
        ]

        loss = -np.mean(np.log(correct_probs + 1e-15))

        return loss

    # Training
    def fit(self, X, y):

        n_samples, n_features = X.shape
        n_classes = 3

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        y_onehot = np.eye(n_classes)[y]

        self.loss_history = []

        print("\nTraining Logistic Regression from scratch...")
        print("Learning Rate:", self.learning_rate)

        for epoch in range(self.epochs):

            # Forward propagation
            z = X @ self.weights + self.bias

            probabilities = self.softmax(z)

            # Calculate loss
            loss = self.compute_loss(y, probabilities)

            loss += (
                self.reg_strength / 2
            ) * np.sum(self.weights ** 2)

            self.loss_history.append(loss)

            # Calculate gradients
            dz = probabilities - y_onehot

            dw = (
                X.T @ dz
            ) / n_samples + self.reg_strength * self.weights

            db = np.mean(dz, axis=0)

            # Gradient descent
            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db

            if epoch % 100 == 0:

                print(
                    f"Epoch {epoch}: Loss = {loss:.4f}"
                )

        print("\nTraining completed!")

    # Prediction
    def predict(self, X):

        z = X @ self.weights + self.bias

        probabilities = self.softmax(z)

        return np.argmax(probabilities, axis=1)


if __name__ == "__main__":

    print("=" * 65)
    print(" SENTINAL AI - LOGISTIC REGRESSION OPTIMIZATION EXPERIMENT")
    print("=" * 65)

    # Load TF-IDF features
    X_train = joblib.load(
        "dataset/processed/features/X_train.pkl"
    )

    X_val = joblib.load(
        "dataset/processed/features/X_val.pkl"
    )

    y_train = joblib.load(
        "dataset/processed/features/y_train.pkl"
    )

    y_val = joblib.load(
        "dataset/processed/features/y_val.pkl"
    )

    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)

    # Learning rates to experiment with
    learning_rates = [2.0, 5.0, 10.0]

    results = []
    trained_models = {}
    loss_curves = {}

    # Experiment
    for lr in learning_rates:

        print("\n" + "=" * 65)
        print("TESTING LEARNING RATE:", lr)
        print("=" * 65)

        model = LogisticRegressionScratch(
            learning_rate=lr,
            epochs=1000,
            reg_strength=0.00002
        )

        # Train model
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

        final_loss = model.loss_history[-1]

        # Store results
        results.append({
            "Learning Rate": lr,
            "Training Accuracy": train_accuracy,
            "Validation Accuracy": val_accuracy,
            "Macro F1": macro_f1,
            "Final Loss": final_loss
        })

        trained_models[lr] = model
        loss_curves[lr] = model.loss_history.copy()

        print("\nTraining Accuracy:", round(train_accuracy, 4))
        print("Validation Accuracy:", round(val_accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))
        print("Final Loss:", round(final_loss, 4))

        print("\nPrediction Distribution:")
        print(np.unique(val_predictions, return_counts=True))

        print("\nClassification Report:")
        print(
            classification_report(
                y_val,
                val_predictions,
                zero_division=0
            )
        )

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, val_predictions))

    # Results table
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("FINAL OPTIMIZATION RESULTS")
    print("=" * 65)

    print(results_df.to_string(index=False))

    # Save experiment results
    results_df.to_csv(
        "evaluation/learning_rate_results.csv",
        index=False
    )

    print("\nResults saved to evaluation/learning_rate_results.csv")

    # Select best model using Macro F1
    best_index = results_df["Macro F1"].idxmax()

    best_lr = results_df.loc[
        best_index,
        "Learning Rate"
    ]

    best_model = trained_models[best_lr]

    # Save best model
    joblib.dump(
        best_model,
        "models/logistic_regression_scratch_best.pkl"
    )

    print("\nBEST LEARNING RATE:", best_lr)

    print(
        "Best Validation Accuracy:",
        results_df.loc[best_index, "Validation Accuracy"]
    )

    print(
        "Best Macro F1:",
        results_df.loc[best_index, "Macro F1"]
    )

    print("\nBest model saved successfully!")

    # Plot loss curves
    plt.figure(figsize=(10, 6))

    for lr in learning_rates:

        plt.plot(
            loss_curves[lr],
            label=f"Learning Rate = {lr}"
        )

    plt.title("Gradient Descent - Learning Rate Comparison")

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "evaluation/learning_rate_comparison.png"
    )

    plt.show()

    print("\nLoss comparison graph saved successfully!")

    # Plot accuracy comparison
    plt.figure(figsize=(9, 5))

    plt.bar(
        results_df["Learning Rate"].astype(str),
        results_df["Validation Accuracy"]
    )

    plt.title("Validation Accuracy vs Learning Rate")

    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy")

    plt.tight_layout()

    plt.savefig(
        "evaluation/learning_rate_accuracy.png"
    )

    plt.show()

    print("\nAccuracy comparison graph saved successfully!")

    print("\nSENTINAL AI OPTIMIZATION EXPERIMENT COMPLETED!")
