import numpy as np
import pandas as pd


LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


def get_model_weights(model):
    """
    Return weights in a consistent shape:
    (number of features, number of classes)
    """

    if hasattr(model, "coef_"):
        weights = model.coef_.T
    else:
        weights = model.weights

    return np.asarray(weights)


def explain_prediction(text, model, vectorizer, top_n=10):
    """
    Explain a sentiment prediction using feature contributions.

    Each contribution represents the effect of a TF-IDF feature
    on one class's raw model score (logit).
    """

    # Validate input
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide valid text.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    # Transform text using the existing TF-IDF vectorizer
    features = vectorizer.transform([text])

    # Retrieve vocabulary and model weights
    feature_names = vectorizer.get_feature_names_out()
    weights = get_model_weights(model)

    # Validate model dimensions
    if weights.shape != (len(feature_names), 3):
        raise ValueError(
            "Model weights do not match the TF-IDF vocabulary "
            "or expected number of classes."
        )

    # Get model bias
    if hasattr(model, "intercept_"):
        bias = np.asarray(model.intercept_)
    else:
        bias = np.asarray(model.bias)

    if bias.shape != (3,):
        raise ValueError("Model bias must contain three class values.")

    # Calculate class scores
    logits = np.asarray(features @ weights + bias).reshape(-1)

    class_scores = {
        LABELS[i]: float(logits[i])
        for i in range(3)
    }

    # Extract non-zero TF-IDF features
    row = features.tocoo()

    contributions = []

    for feature_index, tfidf_value in zip(row.col, row.data):

        word = feature_names[feature_index]

        for class_index in range(3):

            contribution = (
                float(tfidf_value) *
                float(weights[feature_index, class_index])
            )

            contributions.append({
                "Word": word,
                "Sentiment": LABELS[class_index],
                "TF-IDF": float(tfidf_value),
                "Contribution": contribution
            })

    contributions_df = pd.DataFrame(
        contributions,
        columns=[
            "Word",
            "Sentiment",
            "TF-IDF",
            "Contribution"
        ]
    )

    if contributions_df.empty:
        raise ValueError(
            "No recognized vocabulary terms found in the input."
        )

    # Generate explanations for each sentiment
    class_explanations = {}

    for sentiment in LABELS.values():

        class_data = contributions_df[
            contributions_df["Sentiment"] == sentiment
        ].copy()

        # Features increasing the class score
        positive = (
            class_data[class_data["Contribution"] > 0]
            .nlargest(top_n, "Contribution")
            .reset_index(drop=True)
        )

        # Features decreasing the class score
        negative = (
            class_data[class_data["Contribution"] < 0]
            .nsmallest(top_n, "Contribution")
            .reset_index(drop=True)
        )

        class_explanations[sentiment] = {
            "positive": positive,
            "negative": negative
        }

    return {
        "class_scores": class_scores,
        "contributions": contributions_df,
        "class_explanations": class_explanations
    }


def get_prediction_explanation(
    text,
    model,
    vectorizer,
    top_n=10
):
    """
    Return an explanation focused on the model's predicted class.
    """

    result = explain_prediction(
        text,
        model,
        vectorizer,
        top_n
    )

    predicted_sentiment = max(
        result["class_scores"],
        key=result["class_scores"].get
    )

    result["predicted_sentiment"] = predicted_sentiment

    result["predicted_class_explanation"] = (
        result["class_explanations"][predicted_sentiment]
    )

    return result