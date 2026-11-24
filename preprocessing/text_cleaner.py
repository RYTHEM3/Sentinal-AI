import re
import pandas as pd
import os


def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove user mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol but preserve the word
    text = re.sub(r"#", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(input_path, output_path, split):

    df = pd.read_csv(input_path)

    print("\n" + "=" * 50)
    print("Processing:", split.upper())
    print("=" * 50)

    print("Original dataset size:", len(df))

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Identify conflicting labels
    conflicts = (
        df.groupby("clean_text")["label"]
        .nunique()
    )

    conflicting_texts = conflicts[conflicts > 1].index

    print("Conflicting text groups:", len(conflicting_texts))

    if split == "train":

        # Remove texts with conflicting labels
        df = df[
            ~df["clean_text"].isin(conflicting_texts)
        ].copy()

        print("Removed conflicting records.")

        # Remove exact duplicate text-label pairs
        before = len(df)

        df = df.drop_duplicates(
            subset=["clean_text", "label"]
        ).copy()

        print("Duplicate records removed:", before - len(df))

    else:

        print("Records preserved for evaluation.")

    print("Final dataset size:", len(df))

    # Save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print("Saved:", output_path)


if __name__ == "__main__":

    preprocess_dataset(
        "dataset/raw/train.csv",
        "dataset/processed/train.csv",
        "train"
    )

    preprocess_dataset(
        "dataset/raw/validation.csv",
        "dataset/processed/validation.csv",
        "validation"
    )

    preprocess_dataset(
        "dataset/raw/test.csv",
        "dataset/processed/test.csv",
        "test"
    )