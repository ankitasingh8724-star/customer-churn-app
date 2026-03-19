from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

TARGET_COL = "Churn"


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load dataset safely with clear errors.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            f"Expected: data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="utf-8", engine="python")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load dataset: {e}") from e


def clean_telco_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix dataset issues:
      - Convert TotalCharges to numeric
      - Handle blank values
      - Drop rows with missing values
    """
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing required target column '{TARGET_COL}'.")
    if "TotalCharges" not in df.columns:
        raise ValueError("Missing required feature column 'TotalCharges'.")

    df = df.replace(r"^\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Safety for known numeric columns
    for col in ["tenure", "MonthlyCharges", "SeniorCitizen"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    df = df.dropna(axis=0)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def _infer_feature_types(X_raw: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = [c for c in X_raw.columns if c in {"tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"}]
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def encode_features_get_dummies(X_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encode categorical features using get_dummies.
    """
    X_encoded = pd.get_dummies(X_raw, drop_first=False)
    return X_encoded.astype(float), X_encoded.columns.tolist()


def _build_input_ui_schema(
    df_clean: pd.DataFrame,
    input_numeric_columns: List[str],
    input_categorical_columns: List[str],
) -> Dict[str, Any]:
    numeric_schema: Dict[str, Dict[str, float]] = {}
    for col in input_numeric_columns:
        col_min = float(df_clean[col].min())
        col_max = float(df_clean[col].max())

        if col in {"tenure", "SeniorCitizen"}:
            step = 1.0
        elif col in {"MonthlyCharges", "TotalCharges"}:
            step = 0.5
        else:
            step = 0.1

        numeric_schema[col] = {"min": col_min, "max": col_max, "step": step}

    categorical_schema: Dict[str, Dict[str, Any]] = {}
    for col in input_categorical_columns:
        cats = sorted(df_clean[col].astype(str).unique().tolist())
        categorical_schema[col] = {"categories": cats}

    return {"numeric": numeric_schema, "categorical": categorical_schema}


def prepare_training_data(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full training preparation:
      - load dataset safely
      - clean dataset (TotalCharges numeric; blanks -> NaN; drop missing)
      - one-hot encode using get_dummies
      - split into train/test
      - return UI schema + feature columns used by the model
    """
    df = load_dataset(data_path)
    df = clean_telco_dataframe(df)

    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    input_numeric_columns, input_categorical_columns = _infer_feature_types(X_raw)
    input_schema = _build_input_ui_schema(
        df_clean=df,
        input_numeric_columns=input_numeric_columns,
        input_categorical_columns=input_categorical_columns,
    )

    X_encoded, feature_columns = encode_features_get_dummies(X_raw)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "input_numeric_columns": input_numeric_columns,
        "input_categorical_columns": input_categorical_columns,
        "input_schema": input_schema,
        "label_mapping": {0: "No", 1: "Yes"},
    }


def preprocess_customer_input_for_model(
    raw_input: Dict[str, Any],
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Convert raw (unencoded) user input into the exact dummy-encoded columns
    used by the model, then align to training feature columns.
    """
    input_df = pd.DataFrame([raw_input])
    encoded_df = pd.get_dummies(input_df, drop_first=False).astype(float)
    aligned_df = encoded_df.reindex(columns=feature_columns, fill_value=0.0).astype(float)
    return aligned_df

