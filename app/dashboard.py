import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add project root to Python import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.prediction_engine import (
    load_prediction_resources,
    predict_sentiment,
    MODEL_FILES
)

from app.explainability import get_prediction_explanation


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = BASE_DIR / "evaluation"
TEST_DIR = EVAL_DIR / "test_results"

VALIDATION_FILE = EVAL_DIR / "algorithm_comparison.csv"
TEST_FILE = TEST_DIR / "final_test_comparison.csv"


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Sentinal AI | Dashboard",
    page_icon="🤖",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

[data-testid="stMetric"] {
    background-color: #1c2333;
    padding: 18px;
    border-radius: 12px;
}

h1, h2, h3 {
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    validation = pd.read_csv(VALIDATION_FILE)
    test = pd.read_csv(TEST_FILE)

    name_mapping = {
        "Batch GD": "Batch Gradient Descent",
        "Mini-Batch GD": "Mini-Batch Gradient Descent",
        "SGD": "Stochastic Gradient Descent",
        "Newton-CG": "Newton-CG",
        "Golden Section Search": "Golden Section Search"
    }

    validation["Algorithm"] = (
        validation["Algorithm"]
        .astype(str)
        .str.strip()
        .replace(name_mapping)
    )

    test["Algorithm"] = (
        test["Algorithm"]
        .astype(str)
        .str.strip()
    )

    validation = validation.rename(columns={
        "Macro F1": "Validation Macro F1",
        "Training Time": "Validation Training Time"
    })

    test = test.rename(columns={
        "Macro F1": "Test Macro F1",
        "Training Time": "Test Training Time"
    })

    df = pd.merge(
        validation,
        test,
        on="Algorithm",
        how="outer"
    )

    percentage_columns = [
        "Training Accuracy",
        "Validation Accuracy",
        "Validation Macro F1",
        "Test Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Test Macro F1",
        "Weighted F1"
    ]

    for column in percentage_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ) * 100

    for column in [
        "Validation Training Time",
        "Test Training Time"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    for column in df.columns:

        if column != "Algorithm":

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


try:

    df = load_data()

except Exception as error:

    st.error(f"Unable to load project results: {error}")
    st.stop()


# ==================================================
# LOAD PREDICTION RESOURCES
# ==================================================

@st.cache_resource
def get_prediction_resources():

    return load_prediction_resources()


# ==================================================
# HEADER
# ==================================================

st.title("🤖 SENTINAL AI")

st.subheader(
    "Text Sentiment Analysis — Performance Dashboard"
)

st.markdown("""
An interactive machine learning dashboard comparing
six optimization and classification approaches.
""")

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("SENTINAL AI")

st.sidebar.markdown("### Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Performance Comparison",
        "Model Rankings",
        "Training Efficiency",
        "Confusion Matrices",
        "Classification Reports",
        "Final Verdict",
        "Live Prediction"
    ]
)


# ==================================================
# OVERVIEW
# ==================================================

if section == "Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Algorithms", "6")
    col2.metric("Training Samples", "45,570")
    col3.metric("Validation Samples", "2,000")
    col4.metric("Test Samples", "12,284")

    st.divider()

    st.subheader("Sentiment Categories")

    a, b, c = st.columns(3)

    a.error("NEGATIVE")
    b.warning("NEUTRAL")
    c.success("POSITIVE")

    st.divider()

    st.subheader("Algorithms Implemented")

    for algorithm in MODEL_FILES:

        st.write("•", algorithm)

    st.divider()

    st.subheader("Dataset Information")

    st.write("""
    Dataset: TweetEval Sentiment Dataset.

    Task: Classify text into Negative, Neutral and Positive sentiment.

    Features: TF-IDF with unigrams and bigrams.

    Model selection: Validation Macro F1.

    Final evaluation: Previously untouched test split.
    """)

    st.info(
        "The test dataset has a different class distribution "
        "from the training and validation datasets."
    )


# ==================================================
# PERFORMANCE COMPARISON
# ==================================================

elif section == "Performance Comparison":

    st.header("Performance Comparison")

    st.subheader("Validation vs Test Accuracy")

    accuracy_columns = [
        column for column in [
            "Validation Accuracy",
            "Test Accuracy"
        ]
        if column in df.columns
    ]

    accuracy_data = df.melt(
        id_vars="Algorithm",
        value_vars=accuracy_columns,
        var_name="Dataset",
        value_name="Accuracy (%)"
    )

    fig = px.bar(
        accuracy_data,
        x="Algorithm",
        y="Accuracy (%)",
        color="Dataset",
        barmode="group",
        text_auto=".2f",
        title="Accuracy Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Validation vs Test Macro F1")

    f1_columns = [
        column for column in [
            "Validation Macro F1",
            "Test Macro F1"
        ]
        if column in df.columns
    ]

    f1_data = df.melt(
        id_vars="Algorithm",
        value_vars=f1_columns,
        var_name="Dataset",
        value_name="Macro F1 (%)"
    )

    fig = px.bar(
        f1_data,
        x="Algorithm",
        y="Macro F1 (%)",
        color="Dataset",
        barmode="group",
        text_auto=".2f",
        title="Macro F1 Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Complete Results")

    display_columns = [
        "Algorithm",
        "Training Accuracy",
        "Validation Accuracy",
        "Test Accuracy",
        "Validation Macro F1",
        "Test Macro F1"
    ]

    display_columns = [
        column for column in display_columns
        if column in df.columns
    ]

    st.dataframe(
        df[display_columns].round(2),
        use_container_width=True
    )


# ==================================================
# MODEL RANKINGS
# ==================================================

elif section == "Model Rankings":

    st.header("Model Rankings")

    ranked = df.dropna(
        subset=["Test Macro F1"]
    ).sort_values(
        "Test Macro F1",
        ascending=False
    ).reset_index(drop=True)

    ranked.index += 1

    st.subheader("Ranking by Test Macro F1")

    st.dataframe(
        ranked[
            [
                "Algorithm",
                "Test Accuracy",
                "Test Macro F1",
                "Weighted F1"
            ]
        ].round(2),
        use_container_width=True
    )

    fig = px.bar(
        ranked,
        x="Algorithm",
        y="Test Macro F1",
        text_auto=".2f",
        title="Test Macro F1 Ranking"
    )

    st.plotly_chart(fig, use_container_width=True)


# ==================================================
# TRAINING EFFICIENCY
# ==================================================

elif section == "Training Efficiency":

    st.header("Training Efficiency")

    st.write(
        "Training times are taken from recorded experiments. "
        "Missing values are not estimated."
    )

    if "Validation Training Time" in df.columns:

        efficiency = df[
            [
                "Algorithm",
                "Validation Training Time"
            ]
        ].dropna().copy()

        efficiency.rename(
            columns={
                "Validation Training Time":
                "Training Time (seconds)"
            },
            inplace=True
        )

        fig = px.bar(
            efficiency,
            x="Algorithm",
            y="Training Time (seconds)",
            text_auto=".2f",
            title="Training Time Comparison"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            efficiency,
            use_container_width=True
        )

    else:

        st.warning("Training time data is unavailable.")


# ==================================================
# CONFUSION MATRICES
# ==================================================

elif section == "Confusion Matrices":

    st.header("Confusion Matrices")

    matrix_files = {

        "Sklearn Logistic Regression":
            "sklearn_logistic_regression_confusion_matrix.png",

        "Batch Gradient Descent":
            "batch_gradient_descent_confusion_matrix.png",

        "Mini-Batch Gradient Descent":
            "mini_batch_gradient_descent_confusion_matrix.png",

        "Stochastic Gradient Descent":
            "stochastic_gradient_descent_confusion_matrix.png",

        "Newton-CG":
            "newton_cg_confusion_matrix.png",

        "Golden Section Search":
            "golden_section_search_confusion_matrix.png"
    }

    selected_model = st.selectbox(
        "Select Algorithm",
        list(matrix_files.keys())
    )

    image_path = TEST_DIR / matrix_files[selected_model]

    if image_path.exists():

        st.subheader(selected_model)

        st.image(
            str(image_path),
            caption=selected_model + " — Test Confusion Matrix",
            use_container_width=True
        )

    else:

        st.error("Confusion matrix image not found.")


# ==================================================
# CLASSIFICATION REPORTS
# ==================================================

elif section == "Classification Reports":

    st.header("Classification Reports")

    report_files = {

        "Sklearn Logistic Regression":
            "sklearn_logistic_regression_report.txt",

        "Batch Gradient Descent":
            "batch_gradient_descent_report.txt",

        "Mini-Batch Gradient Descent":
            "mini_batch_gradient_descent_report.txt",

        "Stochastic Gradient Descent":
            "stochastic_gradient_descent_report.txt",

        "Newton-CG":
            "newton_cg_report.txt",

        "Golden Section Search":
            "golden_section_search_report.txt"
    }

    selected_model = st.selectbox(
        "Select Algorithm",
        list(report_files.keys())
    )

    report_path = TEST_DIR / report_files[selected_model]

    if report_path.exists():

        report = report_path.read_text(encoding="utf-8")

        st.subheader(selected_model)

        st.code(report, language="text")

    else:

        st.error("Classification report not found.")


# ==================================================
# FINAL VERDICT
# ==================================================

elif section == "Final Verdict":

    st.header("Final Model Evaluation")

    available_models = df.dropna(
        subset=["Test Macro F1"]
    )

    if available_models.empty:

        st.error("Test comparison data is unavailable.")

    else:

        best_overall = available_models.loc[
            available_models["Test Macro F1"].idxmax()
        ]

        custom_models = available_models[
            available_models["Algorithm"]
            != "Sklearn Logistic Regression"
        ]

        best_custom = custom_models.loc[
            custom_models["Test Macro F1"].idxmax()
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.success("🏆 BEST OVERALL MODEL")

            st.subheader(best_overall["Algorithm"])

            st.metric(
                "Test Macro F1",
                f"{best_overall['Test Macro F1']:.2f}%"
            )

            st.metric(
                "Test Accuracy",
                f"{best_overall['Test Accuracy']:.2f}%"
            )

        with col2:

            st.info("🥇 BEST CUSTOM IMPLEMENTATION")

            st.subheader(best_custom["Algorithm"])

            st.metric(
                "Test Macro F1",
                f"{best_custom['Test Macro F1']:.2f}%"
            )

            st.metric(
                "Test Accuracy",
                f"{best_custom['Test Accuracy']:.2f}%"
            )

        st.divider()

        st.subheader("Key Findings")

        st.write("""
        1. Sklearn Logistic Regression achieved the highest
           test Macro F1 score.

        2. Mini-Batch Gradient Descent was the strongest
           custom implementation.

        3. Different optimization methods demonstrate
           different precision and recall trade-offs.

        4. Test performance was lower than validation
           performance, alongside a substantial difference
           in class distribution.
        """)

        st.warning("""
        The test set was evaluated once and was not used
        for model selection or hyperparameter tuning.
        """)

        st.divider()

        st.subheader("Final Comparison")

        st.dataframe(
            available_models[
                [
                    "Algorithm",
                    "Test Accuracy",
                    "Test Macro F1",
                    "Weighted F1"
                ]
            ].sort_values(
                "Test Macro F1",
                ascending=False
            ).round(2),
            use_container_width=True
        )


# ==================================================
# LIVE PREDICTION
# ==================================================

elif section == "Live Prediction":

    st.header("🔮 Live Sentiment Prediction")

    st.markdown("""
    Enter any sentence, tweet, review or comment.
    Select a trained model and discover its predicted sentiment.
    """)

    st.divider()

    # Load models
    try:

        vectorizer, models = get_prediction_resources()

    except Exception as error:

        st.error(f"Unable to load prediction models: {error}")
        st.stop()

    # Initialize session state
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # --------------------------------------------------
    # INPUT
    # --------------------------------------------------

    user_text = st.text_area(
        "Enter text for sentiment analysis",
        placeholder="Example: I absolutely love this product!",
        height=130
    )

    selected_model = st.selectbox(
        "Choose Machine Learning Model",
        list(models.keys())
    )

    col1, col2 = st.columns(2)

    predict_button = col1.button(
        "🚀 Predict Sentiment",
        use_container_width=True
    )

    compare_button = col2.button(
        "⚖️ Compare All Models",
        use_container_width=True
    )

    # --------------------------------------------------
    # INDIVIDUAL PREDICTION
    # --------------------------------------------------

    if predict_button:

        if not user_text.strip():

            st.warning("Please enter some text first.")

        else:

            try:

                model = models[selected_model]

                prediction_result = predict_sentiment(
                    user_text,
                    model,
                    vectorizer
                )

                st.session_state.last_prediction = {
                    "text": user_text,
                    "model": selected_model,
                    "result": prediction_result
                }

                # Clear old explanation when a new prediction is made
                st.session_state.pop(
                    "last_explanation",
                    None
                )

                st.session_state.prediction_history.append({

                    "Text": user_text,
                    "Model": selected_model,
                    "Sentiment": prediction_result["sentiment"],
                    "Score (%)": round(
                        prediction_result["confidence"], 2
                    )

                })

            except Exception as error:

                st.error(f"Prediction failed: {error}")

    # --------------------------------------------------
    # DISPLAY INDIVIDUAL PREDICTION
    # --------------------------------------------------

    if "last_prediction" in st.session_state:

        prediction = st.session_state.last_prediction

        prediction_result = prediction["result"]

        st.divider()

        st.subheader("Prediction Result")

        st.write("**Input Text:**", prediction["text"])

        st.write("**Selected Model:**", prediction["model"])

        sentiment = prediction_result["sentiment"]

        confidence = prediction_result["confidence"]

        col1, col2 = st.columns(2)

        with col1:

            if sentiment == "Positive":

                st.success("😊 POSITIVE SENTIMENT")

            elif sentiment == "Negative":

                st.error("😠 NEGATIVE SENTIMENT")

            else:

                st.warning("😐 NEUTRAL SENTIMENT")

        with col2:

            st.metric(
                "Top-class Model Score",
                f"{confidence:.2f}%"
            )

        st.caption(
            "Scores are softmax/model probabilities and are "
            "not necessarily calibrated confidence estimates."
        )

        st.write(
            "**Cleaned Input:**",
            prediction_result["cleaned_text"]
        )

        st.divider()

        st.subheader("Sentiment Probability Distribution")

        probability_df = pd.DataFrame({

            "Sentiment": list(
                prediction_result["probabilities"].keys()
            ),

            "Probability (%)": list(
                prediction_result["probabilities"].values()
            )

        })

        fig = px.bar(
            probability_df,
            x="Sentiment",
            y="Probability (%)",
            color="Sentiment",
            text_auto=".2f",
            title="Predicted Sentiment Scores"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------------------------
        # MODEL EXPLAINABILITY
        # --------------------------------------------------

        st.divider()

        st.header("🔍 Model Explainability")

        st.markdown("""
        Discover which words influenced the model's
        sentiment prediction and how they affected
        each sentiment class score.
        """)

        explain_button = st.button(
            "🧠 Explain This Prediction",
            use_container_width=True
        )

        if explain_button:

            try:

                with st.spinner(
                    "Analyzing word contributions..."
                ):

                    explanation = get_prediction_explanation(

                        prediction_result["cleaned_text"],

                        models[prediction["model"]],

                        vectorizer,

                        top_n=10
                    )

                    st.session_state.last_explanation = {

                        "text": prediction["text"],

                        "model": prediction["model"],

                        "data": explanation

                    }

            except Exception as error:

                st.error(
                    f"Unable to generate explanation: {error}"
                )

        # --------------------------------------------------
        # DISPLAY EXPLANATION
        # --------------------------------------------------

        if "last_explanation" in st.session_state:

            saved_explanation = st.session_state.last_explanation

            # Ensure explanation belongs to current prediction
            if (
                saved_explanation["text"] == prediction["text"]
                and saved_explanation["model"] == prediction["model"]
            ):

                explanation = saved_explanation["data"]

                predicted_class = explanation["predicted_sentiment"]

                st.divider()

                st.subheader(
                    f"Why did the model predict {predicted_class}?"
                )

                st.info(
                    f"The model predicted **{predicted_class}** "
                    f"using {prediction['model']}."
                )

                # ------------------------------------------
                # CLASS SCORES
                # ------------------------------------------

                st.subheader("Raw Model Class Scores")

                score_columns = st.columns(3)

                for index, (class_name, score) in enumerate(
                    explanation["class_scores"].items()
                ):

                    score_columns[index].metric(
                        class_name,
                        f"{score:.4f}"
                    )

                st.caption(
                    "These are raw model scores (logits), not "
                    "probabilities. A higher score indicates "
                    "a stronger preference for that class."
                )

                st.divider()

                # ------------------------------------------
                # PREDICTED CLASS EXPLANATION
                # ------------------------------------------

                st.subheader(
                    f"Words Influencing {predicted_class} Sentiment"
                )

                predicted_explanation = (
                    explanation["predicted_class_explanation"]
                )

                supporting = predicted_explanation["positive"]

                opposing = predicted_explanation["negative"]

                col1, col2 = st.columns(2)

                with col1:

                    st.success("✅ Supporting Words")

                    if not supporting.empty:

                        st.dataframe(

                            supporting[
                                ["Word", "Contribution"]
                            ].round(4),

                            use_container_width=True,

                            hide_index=True
                        )

                    else:

                        st.info(
                            "No words increased this class score."
                        )

                with col2:

                    st.error("❌ Opposing Words")

                    if not opposing.empty:

                        st.dataframe(

                            opposing[
                                ["Word", "Contribution"]
                            ].round(4),

                            use_container_width=True,

                            hide_index=True
                        )

                    else:

                        st.info(
                            "No words decreased this class score."
                        )

                # ------------------------------------------
                # CONTRIBUTION CHART
                # ------------------------------------------

                st.divider()

                st.subheader("Word Contribution Visualization")

                chart_data = pd.concat(
                    [supporting, opposing],
                    ignore_index=True
                )

                if not chart_data.empty:

                    chart_data["Direction"] = chart_data[
                        "Contribution"
                    ].apply(

                        lambda value:
                        "Supports prediction"
                        if value > 0
                        else "Opposes prediction"

                    )

                    chart_data = chart_data.sort_values(
                        "Contribution",
                        ascending=True
                    )

                    fig = px.bar(

                        chart_data,

                        x="Contribution",

                        y="Word",

                        color="Direction",

                        orientation="h",

                        text_auto=".3f",

                        title=(
                            f"Feature Contributions to "
                            f"{predicted_class} Class Score"
                        )

                    )

                    fig.add_vline(
                        x=0,
                        line_dash="dash"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No non-zero word contributions were found."
                    )

                # ------------------------------------------
                # ALL THREE CLASS EXPLANATIONS
                # ------------------------------------------

                st.divider()

                st.subheader("Class-wise Word Explanations")

                st.markdown("""
                Explore how the same words affect
                Negative, Neutral and Positive class scores.
                """)

                negative_tab, neutral_tab, positive_tab = st.tabs(
                    [
                        "😠 Negative",
                        "😐 Neutral",
                        "😊 Positive"
                    ]
                )

                class_tabs = {

                    "Negative": negative_tab,

                    "Neutral": neutral_tab,

                    "Positive": positive_tab

                }

                for class_name, tab in class_tabs.items():

                    with tab:

                        class_data = explanation[
                            "class_explanations"
                        ][class_name]

                        st.markdown(
                            f"### {class_name} Class"
                        )

                        left, right = st.columns(2)

                        with left:

                            st.markdown("**Supporting words**")

                            if not class_data["positive"].empty:

                                st.dataframe(

                                    class_data["positive"][
                                        ["Word", "Contribution"]
                                    ].round(4),

                                    use_container_width=True,

                                    hide_index=True

                                )

                            else:

                                st.write(
                                    "No positive contributions."
                                )

                        with right:

                            st.markdown("**Opposing words**")

                            if not class_data["negative"].empty:

                                st.dataframe(

                                    class_data["negative"][
                                        ["Word", "Contribution"]
                                    ].round(4),

                                    use_container_width=True,

                                    hide_index=True

                                )

                            else:

                                st.write(
                                    "No negative contributions."
                                )

                # ------------------------------------------
                # EXPLANATION DISCLAIMER
                # ------------------------------------------

                st.divider()

                st.caption("""
                Technical note: Contributions are calculated
                by multiplying each active TF-IDF feature by
                its learned class-specific model weight.

                Positive contributions increase a class's raw
                score, while negative contributions decrease it.

                These explanations describe the behavior of
                the trained linear model. They do not establish
                causation or guarantee that a word has the
                same influence in every context.
                """)

    # --------------------------------------------------
    # COMPARE ALL SIX MODELS
    # --------------------------------------------------

    if compare_button:

        if not user_text.strip():

            st.warning("Please enter some text first.")

        else:

            try:

                comparison_results = []

                for model_name, model in models.items():

                    result = predict_sentiment(
                        user_text,
                        model,
                        vectorizer
                    )

                    comparison_results.append({

                        "Algorithm": model_name,

                        "Sentiment": result["sentiment"],

                        "Top Score (%)": round(
                            result["confidence"], 2
                        ),

                        "Negative (%)": round(
                            result["probabilities"]["Negative"], 2
                        ),

                        "Neutral (%)": round(
                            result["probabilities"]["Neutral"], 2
                        ),

                        "Positive (%)": round(
                            result["probabilities"]["Positive"], 2
                        )

                    })

                comparison_df = pd.DataFrame(
                    comparison_results
                )

                st.session_state.comparison_results = comparison_df

            except Exception as error:

                st.error(f"Comparison failed: {error}")

    # --------------------------------------------------
    # DISPLAY MODEL COMPARISON
    # --------------------------------------------------

    if "comparison_results" in st.session_state:

        st.divider()

        st.subheader("Six-Model Prediction Comparison")

        comparison_df = st.session_state.comparison_results

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(

            comparison_df,

            x="Algorithm",

            y="Top Score (%)",

            color="Sentiment",

            text_auto=".2f",

            title="Model Prediction Score Comparison"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        probability_comparison = comparison_df.melt(

            id_vars="Algorithm",

            value_vars=[
                "Negative (%)",
                "Neutral (%)",
                "Positive (%)"
            ],

            var_name="Sentiment",

            value_name="Probability (%)"

        )

        fig = px.bar(

            probability_comparison,

            x="Algorithm",

            y="Probability (%)",

            color="Sentiment",

            barmode="group",

            title="Probability Comparison Across All Models"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # PREDICTION HISTORY
    # --------------------------------------------------

    st.divider()

    st.subheader("📜 Prediction History")

    if st.session_state.prediction_history:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("🗑️ Clear Prediction History"):

            st.session_state.prediction_history = []

            st.rerun()

    else:

        st.info(
            "Your prediction history will appear here."
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "SENTINAL AI | Text Sentiment Analysis | "
    "Machine Learning & Optimization Project"
)