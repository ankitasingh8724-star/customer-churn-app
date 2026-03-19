from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Tuple

import joblib
import numpy as np

# When running `python src/train_model.py`, the import path includes `src/`,
# so importing `src.*` would fail. Import directly instead.
from preprocessing import prepare_training_data


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _as_numeric_df(X) -> np.ndarray:
    """
    Ensure XGBoost always receives numeric features.
    """
    if hasattr(X, "to_numpy"):
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)
    arr = arr.astype(float, copy=False)
    # Replace any NaNs created by coercion with 0.0
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def evaluate_binary_classifier(model, X_test, y_test) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_pred = model.predict(X_test)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def train_all_models(X_train, y_train, X_test, y_test) -> Tuple[str, Any, Dict[str, Dict[str, float]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, Any] = {}

    for name, model in models.items():
        try:
            if name == "xgboost":
                # XGBoost safety: enforce numeric arrays.
                X_train_np = _as_numeric_df(X_train)
                model.fit(X_train_np, y_train)
                trained_models[name] = model

                X_test_np = _as_numeric_df(X_test)
                metrics = evaluate_binary_classifier(model, X_test_np, y_test)
            else:
                model.fit(X_train, y_train)
                trained_models[name] = model
                metrics = evaluate_binary_classifier(model, X_test, y_test)

            results[name] = metrics
            print(f"[{name}] metrics: {metrics}")
        except Exception as e:
            # Attempt one more time with coerced numeric data for XGBoost only.
            print(f"[{name}] training failed: {e}")
            if name == "xgboost":
                try:
                    X_train_np = _as_numeric_df(X_train)
                    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
                    X_test_np = _as_numeric_df(X_test)
                    model.fit(X_train_np, y_train)
                    trained_models[name] = model
                    metrics = evaluate_binary_classifier(model, X_test_np, y_test)
                    results[name] = metrics
                    print(f"[{name}] metrics (retry): {metrics}")
                except Exception as e2:
                    print(f"[{name}] xgboost retry also failed: {e2}")

    # Select best model:
    # 1) ROC-AUC (higher is better) when available
    # 2) tie-breaker by F1
    def _score(mname: str) -> Tuple[float, float]:
        roc_auc = results.get(mname, {}).get("roc_auc", float("nan"))
        f1 = results.get(mname, {}).get("f1", float("-inf"))
        if np.isnan(roc_auc):
            roc_auc = float("-inf")
        return (roc_auc, f1)

    best_model_name = None
    best = (float("-inf"), float("-inf"))
    for mname in results.keys():
        sc = _score(mname)
        if sc > best:
            best = sc
            best_model_name = mname

    if best_model_name is None:
        raise RuntimeError("All model trainings failed. Cannot select a best model.")

    return best_model_name, trained_models[best_model_name], results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Customer Churn Prediction models.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(_project_root(), "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Path to WA_Fn-UseC_-Telco-Customer-Churn.csv",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    print(f"Using dataset: {args.data_path}")
    try:
        prep = prepare_training_data(
            data_path=args.data_path,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Dataset preparation failed: {e}")
        sys.exit(1)

    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    feature_columns = prep["feature_columns"]

    best_model_name, best_model, results = train_all_models(X_train, y_train, X_test, y_test)
    threshold = 0.5
    metrics = results[best_model_name]

    # Streamlit UI expects these top-level keys:
    # - input_numeric_columns, input_categorical_columns, input_schema
    # - metrics (flat metrics for the chosen best model)
    artifact: Dict[str, Any] = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "metrics": metrics,
        "input_numeric_columns": prep["input_numeric_columns"],
        "input_categorical_columns": prep["input_categorical_columns"],
        "input_schema": prep["input_schema"],
        "label_mapping": prep["label_mapping"],
        # Extra info for debugging/comparison
        "all_model_metrics": results,
    }

    out_path = os.path.join(_project_root(), "models", "best_churn_model.joblib")
    try:
        joblib.dump(artifact, out_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save model artifact to {out_path}: {e}") from e

    print(f"Best model: {best_model_name}")
    print(f"Saved artifact to: {out_path}")
    print(f"Best model metrics: {metrics}")


if __name__ == "__main__":
    main()

