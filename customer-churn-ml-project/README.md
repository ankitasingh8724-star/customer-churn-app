# Customer Churn Prediction

## Project Overview

This project predicts whether a customer will leave a company using machine learning models.

## Features

- Data preprocessing
- Model training
- Churn prediction
- Interactive dashboard

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit

(Additional dependencies: NumPy, joblib, matplotlib; see `requirements.txt`.)

## How to Run

```bash
pip install -r requirements.txt
```

Train a model (expects the Telco churn CSV under `data/`) and save the artifact used by the app:

```bash
python src/train_model.py --data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Start the dashboard:

```bash
streamlit run app.py
```

Optional: use a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Live Demo

(Add your Streamlit link here)

## Dataset

Default path:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

If the file is missing, training fails with a clear error.

## Training pipeline (summary)

1. Load data with validation; fix `TotalCharges`, drop rows with missing values
2. One-hot encode categoricals
3. Train Logistic Regression, Random Forest, and XGBoost; pick best by ROC-AUC (F1 tie-break)
4. Save `models/best_churn_model.joblib` for the Streamlit app
