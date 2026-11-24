import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class MiniBatchLogisticRegression:

    def __init__(
        self,
        learning_rate=5.0,
        epochs=100,
        batch_size=256,
        reg_strength=0.00002
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_strength = reg_strength

        self.weights = None
        self.bias = None
        self.loss_history = []

    # Softmax
    def softmax(self, z):

        z = z - np.max(z, axis=1, keepdims=True)

        exp_z = np.exp(z)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Cross Entropy Loss
    def compute_loss(self, y, probabilities):

        n_samples = len(y)

        correct_probs = probabilities[
            np.arange(n_samples),
            y
        ]

        return -np.mean(np.log(correct_probs + 1e-15))

    # Training
    def fit(self, X, y):

        n_samples, n_features = X.shape
        n_classes = 3

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        self.loss_history = []

        rng = np.random.default_rng(42)

        print("\nTraining Mini-Batch Logistic Regression...")
        print("Learning Rate:", self.learning_rate)
        print("Batch Size:", self.batch_size)

        start_time = time.time()

        for epoch in range(self.epochs):

            # Shuffle training data
            indices = rng.permutation(n_samples)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Process mini-batches
            for start in range(0, n_samples, self.batch_size):

                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                batch_samples = len(y_batch)

                # One-hot encoding
                y_onehot = np.eye(n_classes)[y_batch]

                # Forward propagation
                z = X_batch @ self.weights + self.bias

                probabilities = self.softmax(z)

                # Calculate gradients
                dz = probabilities - y_onehot

                dw = (
                    X_batch.T @ dz
                ) / batch_samples + self.reg_strength * self.weights

                db = np.mean(dz, axis=0)

                # Update parameters
                self.weights -= self.learning_rate * dw

                self.bias -= self.learning_rate * db

            # Calculate full training loss after each epoch
            full_z = X @ self.weights + self.bias

            full_probabilities = self.softmax(full_z)

            loss = self.compute_loss(y, full_probabilities)

            loss += (
                self.reg_strength / 2
            ) * np.sum(self.weights ** 2)

            self.loss_history.append(loss)

            if epoch % 10 == 0:

                print(
                    f"Epoch {epoch}: Loss = {loss:.4f}"
                )

        self.training_time = time.time() - start_time

        print("\nTraining completed!")
        print("Training Time:", round(self.training_time, 2), "seconds")

    # Prediction
    def predict(self, X):

        z = X @ self.weights + self.bias

        probabilities = self.softmax(z)

        return np.argmax(probabilities, axis=1)


if __name__ == "__main__":

    print("=" * 65)
    print(" SENTINAL AI - MINI-BATCH GRADIENT DESCENT")
    print("=" * 65)

    # Load TF-IDF features
    X_train = joblib.load(
        "dataset/processed/features/X_train.pkl"
    )

    X_val = joblib.load(
        "dataset/processed/features/X_val.pkl"
    )

    y_train = np.asarray(
        joblib.load("dataset/processed/features/y_train.pkl"),
        dtype=int
    )

    y_val = np.asarray(
        joblib.load("dataset/processed/features/y_val.pkl"),
        dtype=int
    )

    # Initialize model
    model = MiniBatchLogisticRegression(
        learning_rate=5.0,
        epochs=100,
        batch_size=256,
        reg_strength=0.00002
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

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print("Training Accuracy:", round(train_accuracy, 4))
    print("Validation Accuracy:", round(val_accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Final Loss:", round(model.loss_history[-1], 4))

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
        "models/mini_batch_logistic_regression.pkl"
    )

    # Save evaluation results
    results = pd.DataFrame([{
        "Optimization": "Mini-Batch Gradient Descent",
        "Learning Rate": model.learning_rate,
        "Batch Size": model.batch_size,
        "Epochs": model.epochs,
        "Training Accuracy": train_accuracy,
        "Validation Accuracy": val_accuracy,
        "Macro F1": macro_f1,
        "Final Loss": model.loss_history[-1],
        "Training Time": model.training_time
    }])

    results.to_csv(
        "evaluation/mini_batch_results.csv",
        index=False
    )

    # Plot loss
    plt.figure(figsize=(9, 5))

    plt.plot(model.loss_history)

    plt.title("Mini-Batch Gradient Descent - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "evaluation/mini_batch_loss.png"
    )

    plt.show()

    print("\nModel saved successfully!")
    print("Evaluation results saved!")
    print("Loss graph saved!")

    print("\nSENTINAL AI MINI-BATCH TRAINING COMPLETED!")