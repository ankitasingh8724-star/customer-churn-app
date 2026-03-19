# from __future__ import annotations (disabled duplicate)

import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_customer_input_for_model


def load_model_artifact(model_path: str) -> Dict[str, Any]:
    """
    Load the trained model artifact created by `src/train_model.py`.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at: {model_path}\n"
            "Run training first to generate `models/best_churn_model.joblib`."
        )

    try:
        artifact = joblib.load(model_path)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load model artifact: {e}") from e

    if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError("Invalid artifact format: expected dict with keys `model` and `feature_columns`.")

    return artifact


def _predict_proba(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])
    if hasattr(model, "decision_function"):
        from scipy.special import expit

        scores = model.decision_function(X)
        return float(expit(scores[0]))
    raise ValueError("Model does not support predict_proba or decision_function.")


def predict_churn(input_data: Dict[str, Any], artifact: Dict[str, Any]) -> Tuple[float, int]:
    """
    Predict churn probability and class label (0/1).
    """
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact.get("threshold", 0.5))

    X = preprocess_customer_input_for_model(raw_input=input_data, feature_columns=feature_columns).astype(float)
    proba = _predict_proba(model, X)
    pred = int(proba >= threshold)
    return proba, pred


def predict_churn_proba(
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Streamlit-friendly wrapper (kept for backward compatibility).
    """
    proba, pred = predict_churn(input_data=input_data, artifact=artifact)
    return proba, pred, {}


def explain_with_shap(
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
    top_k: int = 15,
) -> Dict[str, Any]:
    """
    Best-effort SHAP explanation.

    If SHAP is not installed or explanation fails, return empty results instead of crashing.
    """
    try:
        import shap  # type: ignore
    except Exception as e:
        return {"top_features": [], "top_shap_values": [], "error": f"SHAP import failed: {e}"}

    try:
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]

        X = preprocess_customer_input_for_model(raw_input=input_data, feature_columns=feature_columns).astype(float)
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        vals = np.array(shap_values.values)[0]

        idx = np.argsort(np.abs(vals))[::-1][: int(top_k)]
        top_features = [feature_columns[i] for i in idx]
        top_vals = [float(vals[i]) for i in idx]
        return {"top_features": top_features, "top_shap_values": top_vals}
    except Exception as e:
        return {"top_features": [], "top_shap_values": [], "error": f"SHAP explanation failed: {e}"}

# from __future__ import annotations (disabled duplicate)

import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_customer_input_for_model


def load_model_artifact(model_path: str) -> Dict[str, Any]:
    """
    Load the trained model artifact created by `src/train_model.py`.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at: {model_path}\n"
            "Run training first to generate `models/best_churn_model.joblib`."
        )

    try:
        artifact = joblib.load(model_path)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load model artifact: {e}") from e

    if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError("Invalid artifact format: expected dict with keys `model` and `feature_columns`.")

    return artifact


def _predict_proba(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])
    if hasattr(model, "decision_function"):
        from scipy.special import expit

        scores = model.decision_function(X)
        return float(expit(scores[0]))
    raise ValueError("Model does not support predict_proba or decision_function.")


def predict_churn(input_data: Dict[str, Any], artifact: Dict[str, Any]) -> Tuple[float, int]:
    """
    Predict churn probability and class label (0/1).
    """
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact.get("threshold", 0.5))

    X = preprocess_customer_input_for_model(raw_input=input_data, feature_columns=feature_columns).astype(float)
    proba = _predict_proba(model, X)
    pred = int(proba >= threshold)
    return proba, pred


def predict_churn_proba(
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Streamlit-friendly wrapper (kept for backward compatibility).
    """
    proba, pred = predict_churn(input_data=input_data, artifact=artifact)
    return proba, pred, {}


def explain_with_shap(
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
    top_k: int = 15,
) -> Dict[str, Any]:
    """
    Best-effort SHAP explanation.

    App enables SHAP via a checkbox; if SHAP is not installed or fails, we return
    empty results rather than crashing.
    """
    try:
        import shap  # type: ignore
    except Exception as e:
        return {"top_features": [], "top_shap_values": [], "error": f"SHAP import failed: {e}"}

    try:
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        X = preprocess_customer_input_for_model(raw_input=input_data, feature_columns=feature_columns).astype(float)

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        vals = np.array(shap_values.values)[0]

        idx = np.argsort(np.abs(vals))[::-1][: int(top_k)]
        top_features = [feature_columns[i] for i in idx]
        top_vals = [float(vals[i]) for i in idx]
        return {"top_features": top_features, "top_shap_values": top_vals}
    except Exception as e:
        return {"top_features": [], "top_shap_values": [], "error": f"SHAP explanation failed: {e}"}

# from __future__ import annotations (disabled duplicate)

import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_customer_input_for_model


def load_model_artifact(model_path: str) -> Dict[str, Any]:
    """
    Load the saved joblib artifact created by `src/train_model.py`.
    Resolves relative paths against the project root for reliability.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at: {model_path}\n"
            "Run training first to generate `models/best_churn_model.joblib`."
        )

    try:
        artifact = joblib.load(model_path)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load model artifact: {e}") from e

    if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError(
            "Invalid artifact format: expected a dict with keys "
            "`model` and `feature_columns`."
        )
    return artifact


def _predict_proba(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        return float(proba[0])
    if hasattr(model, "decision_function"):
        from scipy.special import expit

        scores = model.decision_function(X)
        return float(expit(scores[0]))
    raise ValueError("Model does not support predict_proba or decision_function.")


def predict_churn(input_data: Dict[str, Any], artifact: Dict[str, Any]) -> Tuple[float, int]:
    """
    Predict churn probability and class label (0/1).
    """
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact.get("threshold", 0.5))

    X = preprocess_customer_input_for_model(raw_input=input_data, feature_columns=feature_columns)
    X = X.astype(float)

    proba = _predict_proba(model, X)
    pred = int(proba >= threshold)
    return proba, pred


def predict_churn_proba(
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Backwards-compatible wrapper for the Streamlit UI in `app.py`.
    """
    proba, pred = predict_churn(input_data=input_data, artifact=artifact)
    return proba, pred, {}


def explain_with_shap(
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
    top_k: int = 15,
) -> Dict[str, Any]:
    """
    Optional SHAP explanation (best-effort).

    The app will display this only when the user enables the checkbox.
    If SHAP isn't available or explanation fails, we return empty results
    (so the app won't crash).
    """
    try:
        import shap  # lazy import
    except Exception as e:
        return {"top_features": [], "top_shap_values": [], "error": f"SHAP import failed: {e}"}

    try:
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]

        X = preprocess_customer_input_for_model(raw_input=input_data, feature_columns=feature_columns)
        X = X.astype(float)

        # Generic SHAP explainer; may be slower but keeps logic simple.
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        vals = shap_values.values[0]

        idx = np.argsort(np.abs(vals))[::-1][: int(top_k)]
        top_features = [feature_columns[i] for i in idx]
        top_vals = [float(vals[i]) for i in idx]
        return {"top_features": top_features, "top_shap_values": top_vals}
    except Exception as e:
        return {"top_features": [], "top_shap_values": [], "error": f"SHAP explanation failed: {e}"}

# from __future__ import annotations (disabled duplicate)

import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_customer_input_for_model


def load_model_artifact(model_path: str) -> Dict[str, Any]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at: {model_path}\n"
            f"Run training first to generate the joblib file."
        )
    try:
        artifact = joblib.load(model_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError("Invalid artifact format: expected a dict with key 'model'.")
        return artifact
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load model artifact: {e}") from e


def _coerce_prediction_proba(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        return float(proba[0])

    # Fallback: decision_function -> sigmoid
    if hasattr(model, "decision_function"):
        from scipy.special import expit

        scores = model.decision_function(X)
        return float(expit(scores[0]))

    raise ValueError("Model does not support predict_proba or decision_function.")


def predict_churn(input_data: Dict[str, Any], artifact: Dict[str, Any]) -> Tuple[float, int]:
    """
    Predict churn probability and class label (0/1).
    """
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact.get("threshold", 0.5))

    # Preprocess raw user input to dummy-encoded model features
    X = preprocess_customer_input_for_model(
        raw_input=input_data,
        feature_columns=feature_columns,
    )

    # Ensure numeric for XGBoost-like estimators
    X = X.astype(float)

    proba = _coerce_prediction_proba(model, X)
    pred = int(proba >= threshold)
    return proba, pred


def predict_churn_proba(
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Backwards-compatible wrapper for the Streamlit UI.
    """
    proba, pred = predict_churn(input_data=input_data, artifact=artifact)
    return proba, pred, {}


def explain_with_shap(
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
    top_k: int = 15,
) -> Dict[str, Any]:
    """
    Optional SHAP explanation (best-effort).
    If SHAP fails (missing deps or model incompatibility), return empty results
    rather than crashing the app.
    """
    try:
        import shap  # lazy import so the app still works without SHAP
    except Exception as e:
        return {"top_features": [], "top_shap_values": [], "error": f"SHAP import failed: {e}"}

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X = preprocess_customer_input_for_model(raw_input=input_data, feature_columns=feature_columns)
    X = X.astype(float)

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        vals = shap_values.values[0]

        # pick top_k by absolute impact
        idx = np.argsort(np.abs(vals))[::-1][:top_k]
        top_features = [feature_columns[i] for i in idx]
        top_vals = [float(vals[i]) for i in idx]
        return {"top_features": top_features, "top_shap_values": top_vals}
    except Exception as e:
        return {"top_features": [], "top_shap_values": [], "error": f"SHAP explanation failed: {e}"}

# from __future__ import annotations (disabled duplicate)

import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


def load_model_artifact(model_path: str) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at: {model_path}\n"
            "Train the model first using `python src/train_model.py --data_path ...`."
        )
    artifact = joblib.load(model_path)
    if "model_pipeline" not in artifact:
        raise ValueError("Invalid model artifact: missing `model_pipeline`.")
    return artifact


def _build_input_dataframe(input_data: Dict[str, Any], input_feature_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in input_feature_columns if c not in input_data]
    if missing:
        raise ValueError(f"Missing required input features: {missing}")

    row = {c: input_data[c] for c in input_feature_columns}
    return pd.DataFrame([row])


def predict_churn_proba(
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Returns: (probability_of_churn, predicted_label_0_1, details)
    """
    model_pipeline = artifact["model_pipeline"]
    threshold = float(artifact.get("threshold", 0.5))
    input_feature_columns = artifact["input_feature_columns"]

    X_input = _build_input_dataframe(input_data, input_feature_columns)
    proba = model_pipeline.predict_proba(X_input)[:, 1][0]
    pred = int(proba >= threshold)

    details = {
        "threshold": threshold,
        "probability_churn": float(proba),
        "predicted_label": pred,
        "label_mapping": artifact.get("label_mapping", {0: "No", 1: "Yes"}),
    }
    return float(proba), pred, details


def explain_with_shap(
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
    top_k: int = 15,
) -> Dict[str, Any]:
    """
    SHAP explanations for the single input row.
    Produces top-k feature contributions for a clean UI.
    """
    shap = __import__("shap")  # Lazy import so app can still run without shap.

    model_pipeline = artifact["model_pipeline"]
    model = model_pipeline.named_steps["model"]
    threshold = float(artifact.get("threshold", 0.5))  # kept for reference
    feature_names_out = artifact.get("feature_names_out", [])
    input_feature_columns = artifact["input_feature_columns"]

    X_input_raw = _build_input_dataframe(input_data, input_feature_columns)

    # pipeline = [feature_engineering -> preprocess -> model]
    pre_model = model_pipeline[:-1]  # everything before estimator
    X_processed = pre_model.transform(X_input_raw)
    if isinstance(X_processed, np.ndarray) is False:
        X_processed = np.array(X_processed)

    background_processed = artifact.get("background_processed")
    if background_processed is None:
        # Fallback: use current processed input as background.
        background_processed = X_processed

    model_name = artifact.get("best_model_name", "")

    if model_name.lower() in ["random_forest", "xgboost"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        # For binary classification, shap_values can be a list: [class0, class1]
        if isinstance(shap_values, list):
            shap_values_for_churn = shap_values[1]
        else:
            shap_values_for_churn = shap_values
        row_shap = np.array(shap_values_for_churn)[0]
    else:
        # Linear model explanation
        explainer = shap.LinearExplainer(model, background_processed)
        shap_values = explainer.shap_values(X_processed)
        if isinstance(shap_values, list):
            shap_values_for_churn = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values_for_churn = shap_values
        row_shap = np.array(shap_values_for_churn)[0]

    # Pick top-k by absolute impact.
    if feature_names_out and len(feature_names_out) == row_shap.shape[0]:
        names = np.array(feature_names_out)
    else:
        names = np.array([f"f{i}" for i in range(row_shap.shape[0])])

    k = int(min(top_k, row_shap.shape[0]))
    top_idx = np.argsort(np.abs(row_shap))[::-1][:k]
    top_features = names[top_idx].tolist()
    top_values = row_shap[top_idx].astype(float).tolist()

    proba, pred, details = predict_churn_proba(input_data, artifact)

    return {
        "probability_churn": proba,
        "predicted_label": pred,
        "threshold": threshold,
        "top_features": top_features,
        "top_shap_values": top_values,
        "raw_row_shap_values": row_shap,
        "feature_names_out": feature_names_out,
    }


# ==========================================================
# Authoritative inference implementations (override legacy code)
# ==========================================================

def _predict_proba(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        return float(proba[0])
    if hasattr(model, "decision_function"):
        from scipy.special import expit

        scores = model.decision_function(X)
        return float(expit(scores[0]))
    raise ValueError("Model does not support predict_proba or decision_function.")


def load_model_artifact(model_path: str) -> Dict[str, Any]:  # type: ignore[override]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at: {model_path}\n"
            "Run training first to generate `models/best_churn_model.joblib`."
        )

    try:
        artifact = joblib.load(model_path)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load model artifact: {e}") from e

    if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError("Invalid artifact format: expected keys `model` and `feature_columns`.")
    return artifact


def predict_churn_proba(  # type: ignore[override]
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Tuple[float, int, Dict[str, Any]]:
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact.get("threshold", 0.5))

    X = preprocess_customer_input_for_model(
        raw_input=input_data,
        feature_columns=feature_columns,
    ).astype(float)

    proba = _predict_proba(model, X)
    pred = int(proba >= threshold)
    return proba, pred, {}


def explain_with_shap(  # type: ignore[override]
    input_data: Dict[str, Any],
    artifact: Dict[str, Any],
    top_k: int = 15,
) -> Dict[str, Any]:
    """
    Best-effort SHAP explanation.
    If anything fails, return an empty plot input so the app doesn't crash.
    """
    try:
        import shap  # lazy import
    except Exception as e:
        return {"top_features": [], "top_shap_values": [], "error": f"SHAP import failed: {e}"}

    try:
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        X = preprocess_customer_input_for_model(raw_input=input_data, feature_columns=feature_columns).astype(float)

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        vals = shap_values.values[0]

        idx = np.argsort(np.abs(vals))[::-1][: int(top_k)]
        top_features = [feature_columns[i] for i in idx]
        top_vals = [float(vals[i]) for i in idx]
        return {"top_features": top_features, "top_shap_values": top_vals}
    except Exception as e:
        return {"top_features": [], "top_shap_values": [], "error": f"SHAP explanation failed: {e}"}

