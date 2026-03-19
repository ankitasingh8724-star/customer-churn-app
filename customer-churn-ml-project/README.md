# Customer Churn Prediction System

End-to-end, production-ready ML project for Telco customer churn prediction.

## Dataset

Place the dataset at:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Setup

From the project directory:

```bash
cd customer-churn-ml-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the Models

```bash
python3 src/train_model.py --data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

What the training pipeline does:

1. Safe dataset loading with clear errors
2. Data fixes:
   - Converts `TotalCharges` to numeric
   - Treats blank values as missing
   - Drops rows with missing values
3. One-hot encodes categorical features using `pd.get_dummies`
4. Splits into train/test
5. Trains:
   - Logistic Regression
   - Random Forest
   - XGBoost (with numeric-feature safety)
6. Compares models using ROC-AUC (primary) and F1 (tie-break)
7. Saves the best model with `joblib` to:
   - `models/best_churn_model.joblib`

## Run the Streamlit App

```bash
streamlit run app.py
```

The app collects customer details and predicts churn probability + churn/not-churn using the saved model.

## Notes

- If `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` is missing, training will fail with a clear message.

