import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer


print("=" * 55)
print("       SENTINAL AI - TF-IDF FEATURE EXTRACTION")
print("=" * 55)


# Load processed datasets
train_df = pd.read_csv("dataset/processed/train.csv")
val_df = pd.read_csv("dataset/processed/validation.csv")
test_df = pd.read_csv("dataset/processed/test.csv")


# Extract text
X_train_text = train_df["clean_text"].fillna("")
X_val_text = val_df["clean_text"].fillna("")
X_test_text = test_df["clean_text"].fillna("")


# Extract labels
y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]


# Initialize TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


print("\nFitting TF-IDF on training data...")

# Fit ONLY on training data
X_train = vectorizer.fit_transform(X_train_text)

# Transform validation and testing data
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)


# Display dimensions
print("\nFEATURE MATRIX DIMENSIONS")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


print("\nVocabulary size:", len(vectorizer.vocabulary_))


# Save feature extractor
os.makedirs("models", exist_ok=True)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


# Save feature matrices
os.makedirs("dataset/processed/features", exist_ok=True)

joblib.dump(X_train, "dataset/processed/features/X_train.pkl")
joblib.dump(X_val, "dataset/processed/features/X_val.pkl")
joblib.dump(X_test, "dataset/processed/features/X_test.pkl")

joblib.dump(y_train, "dataset/processed/features/y_train.pkl")
joblib.dump(y_val, "dataset/processed/features/y_val.pkl")
joblib.dump(y_test, "dataset/processed/features/y_test.pkl")


print("\nTF-IDF FEATURE EXTRACTION COMPLETED!")