# Sentinal AI – Text Sentiment Analysis

An interactive machine learning application that classifies text into three sentiment categories: Negative, Neutral, and Positive.

Built using Python, Streamlit, TF-IDF feature extraction, Logistic Regression, and multiple optimization algorithms.

## Features

- Real-time sentiment prediction.
- Six trained machine learning models.
- Compare predictions across all six models.
- Model performance comparison dashboard.
- Model rankings based on evaluation metrics.
- Confusion matrices and classification reports.
- Feature-level explainability showing words supporting or opposing predicted sentiment.
- Interactive visualizations using Plotly.

## Machine Learning Models

1. Scikit-learn Logistic Regression
2. Batch Gradient Descent
3. Mini-Batch Gradient Descent
4. Stochastic Gradient Descent
5. Newton-CG
6. Golden Section Search

## Dataset

This project uses the TweetEval sentiment dataset.

The classification labels are:

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

The dataset is divided into training, validation, and test sets.

The test set is evaluated separately from model selection.

## Technologies Used

- Python 3.12
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Joblib
- Streamlit
- Plotly
- Matplotlib

## Project Structure

```text
Sentinal-AI/
│
├── app/
│   ├── dashboard.py
│   ├── prediction_engine.py
│   └── explainability.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── logistic_regression_scratch_best.pkl
│   ├── mini_batch_logistic_regression.pkl
│   ├── stochastic_logistic_regression.pkl
│   ├── newtons_method_model.pkl
│   ├── golden_section_search.pkl
│   └── tfidf_vectorizer.pkl
│
├── optimization/
├── preprocessing/
├── feature_extraction/
│
├── evaluation/
│   ├── algorithm_comparison.csv
│   └── test_results/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd Sentinal-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app/dashboard.py
```

Open the local URL displayed in your terminal.

## Evaluation

The project evaluates models using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score
- Weighted F1 Score
- Training Time
- Confusion Matrices

The final test evaluation is kept separate from validation-based model selection.

## Explainability

Sentinal AI provides feature-level explanations using TF-IDF feature contributions and model weights.

It identifies words that increase or decrease the raw score of each sentiment class.

These contributions explain model behavior; they should not be interpreted as causal explanations.

## Deployment

The application is designed for deployment using Streamlit Community Cloud.

## Disclaimer

This project classifies sentiment expressed in text. It does not determine a person's actual emotions, intentions, or psychological state.

---

**Sentinal AI**  
Text Sentiment Analysis using Machine Learning and Optimization Algorithms.