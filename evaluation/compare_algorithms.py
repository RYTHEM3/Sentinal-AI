
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent

# Results from all five optimization algorithms
results = [
    {
        "Algorithm": "Batch Gradient Descent",
        "Training Accuracy": 0.667215,
        "Validation Accuracy": 0.6375,
        "Macro F1": 0.544601,
        "Final Loss": 0.774624,
        "Training Time": None
    },
    {
        "Algorithm": "Mini-Batch Gradient Descent",
        "Training Accuracy": 0.719377,
        "Validation Accuracy": 0.6245,
        "Macro F1": 0.610002,
        "Final Loss": 0.762928,
        "Training Time": 5.062774
    },
    {
        "Algorithm": "Stochastic Gradient Descent",
        "Training Accuracy": 0.667939,
        "Validation Accuracy": 0.6330,
        "Macro F1": 0.551154,
        "Final Loss": 0.782160,
        "Training Time": 49.590179
    },
    {
        "Algorithm": "Newton-CG",
        "Training Accuracy": 0.665745,
        "Validation Accuracy": 0.6345,
        "Macro F1": 0.533170,
        "Final Loss": 0.780221,
        "Training Time": 0.494606
    },
    {
        "Algorithm": "Golden Section Search",
        "Training Accuracy": 0.649989,
        "Validation Accuracy": 0.6105,
        "Macro F1": 0.567972,
        "Final Loss": 0.790634,
        "Training Time": 40.709779
    }
]

df = pd.DataFrame(results)

# Sort by Macro F1
df = df.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)

df.insert(0, "Rank", range(1, len(df) + 1))

# Save comparison CSV
csv_path = BASE_DIR / "algorithm_comparison.csv"
df.to_csv(csv_path, index=False)

print("\n========== MASTER ALGORITHM COMPARISON ==========")
print(df.to_string(index=False))

# Chart 1: Validation Accuracy and Macro F1
plt.figure(figsize=(12, 6))

plot_df = df.set_index("Algorithm")[
    ["Validation Accuracy", "Macro F1"]
] * 100

plot_df.plot(kind="bar", figsize=(12, 6))

plt.title("Optimization Algorithm Performance Comparison")
plt.ylabel("Score (%)")
plt.xlabel("Optimization Algorithm")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig(
    BASE_DIR / "algorithm_comparison.png",
    dpi=300
)

plt.close()

# Chart 2: Training Time
time_df = df.dropna(subset=["Training Time"]).sort_values(
    by="Training Time"
)

plt.figure(figsize=(11, 6))

plt.bar(
    time_df["Algorithm"],
    time_df["Training Time"]
)

plt.title("Optimization Algorithm Training Time")
plt.ylabel("Training Time (seconds)")
plt.xlabel("Optimization Algorithm")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "algorithm_training_time.png",
    dpi=300
)

plt.close()

print("\nFiles generated successfully:")
print(csv_path)
print(BASE_DIR / "algorithm_comparison.png")
print(BASE_DIR / "algorithm_training_time.png")
