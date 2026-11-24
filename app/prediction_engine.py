
import sys
import importlib
import joblib
import numpy as np

from pathlib import Path
from preprocessing.text_cleaner import clean_text


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"


# ==================================================
# MODEL CONFIGURATION
# ==================================================

MODEL_FILES = {
    "Sklearn Logistic Regression":
        "logistic_regression.pkl",

    "Batch Gradient Descent":
        "logistic_regression_scratch_best.pkl",

    "Mini-Batch Gradient Descent":
        "mini_batch_logistic_regression.pkl",

    "Stochastic Gradient Descent":
        "stochastic_logistic_regression.pkl",

    "Newton-CG":
        "newtons_method_model.pkl",

    "Golden Section Search":
        "golden_section_search.pkl"
}


CLASS_LOCATIONS = {
    "GoldenSectionLogisticRegression":
        "optimization.golden_section_search",

    "LogisticRegressionScratch":
        "optimization.logistic_regression_scratch",

    "MiniBatchLogisticRegression":
        "optimization.mini_batch_gradient_descent",

    "NewtonLogisticRegression":
        "optimization.newtons_method",

    "StochasticLogisticRegression":
        "optimization.stochastic_gradient_descent"
}


LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ==================================================
# REGISTER CUSTOM MODEL CLASSES
# ==================================================

def register_model_classes():

    sys.path.insert(0, str(BASE_DIR))

    main_module = sys.modules["__main__"]

    for class_name, module_name in CLASS_LOCATIONS.items():

        module = importlib.import_module(module_name)

        model_class = getattr(module, class_name)

        setattr(
            main_module,
            class_name,
            model_class
        )


# ==================================================
# LOAD MODELS AND VECTORIZER
# ==================================================

def load_prediction_resources():

    register_model_classes()

    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

    vectorizer = joblib.load(vectorizer_path)

    models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        models[model_name] = joblib.load(model_path)

    return vectorizer, models


# ==================================================
# CALCULATE PROBABILITIES
# ==================================================

def get_probabilities(model, features):

    # Sklearn Logistic Regression
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(features)

    else:

        # Custom optimization models
        logits = features @ model.weights + model.bias

        probabilities = model.softmax(logits)

    probabilities = np.asarray(probabilities)

    return probabilities[0]


# ==================================================
# PREDICT SENTIMENT
# ==================================================

def predict_sentiment(text, model, vectorizer):

    if not isinstance(text, str) or not text.strip():

        raise ValueError("Please enter some text.")

    # Apply original preprocessing
    cleaned_text = clean_text(text)

    if not cleaned_text:

        raise ValueError(
            "No usable text remains after preprocessing."
        )

    # Transform using original fitted TF-IDF vectorizer
    features = vectorizer.transform([cleaned_text])

    # Calculate probabilities
    probabilities = get_probabilities(
        model,
        features
    )

    # Predicted class
    predicted_class = int(np.argmax(probabilities))

    sentiment = LABELS[predicted_class]

    confidence = float(probabilities[predicted_class])

    probability_results = {
        LABELS[index]: float(probabilities[index]) * 100
        for index in range(3)
    }

    return {
        "sentiment": sentiment,
        "confidence": confidence * 100,
        "probabilities": probability_results,
        "cleaned_text": cleaned_text
    }
