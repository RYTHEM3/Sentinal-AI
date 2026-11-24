import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 55)
print("          SENTINAL AI - EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# Load training dataset
df = pd.read_csv("dataset/raw/train.csv")

# Label mapping
label_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

df["sentiment"] = df["label"].map(label_mapping)

# 1. Display sample records
print("\n1. SAMPLE RECORDS")
print(df[["text", "sentiment"]].head(10).to_string(index=False))

# 2. Dataset information
print("\n2. DATASET INFORMATION")
print("Total records:", len(df))
print("Missing values:\n", df.isnull().sum())
print("Duplicate records:", df.duplicated(subset=["text"]).sum())

# 3. Calculate text length
df["text_length"] = df["text"].astype(str).apply(len)

print("\n3. TEXT LENGTH STATISTICS")
print(df["text_length"].describe())

# 4. Sentiment distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="sentiment",
    order=["Negative", "Neutral", "Positive"]
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("notebooks/sentiment_distribution.png")
plt.show()

# 5. Text length distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="text_length",
    bins=50
)

plt.title("Tweet Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("notebooks/text_length_distribution.png")
plt.show()

# 6. Average text length by sentiment
print("\n4. AVERAGE TEXT LENGTH BY SENTIMENT")

print(
    df.groupby("sentiment")["text_length"]
    .mean()
    .round(2)
)

print("\nEDA COMPLETED SUCCESSFULLY!")
