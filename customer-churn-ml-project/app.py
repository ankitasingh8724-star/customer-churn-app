from __future__ import annotations

import os
from typing import Any, Dict

mpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl")
os.makedirs(mpl_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_dir)

import matplotlib.pyplot as plt
import streamlit as st

from src.predict import explain_with_shap, load_model_artifact, predict_churn_proba


st.set_page_config(page_title="Customer Churn Prediction System", layout="wide")


@st.cache_resource
def get_artifact() -> Dict[str, Any]:
    model_path = os.path.join("models", "best_churn_model.joblib")
    return load_model_artifact(model_path)


def _render_inputs(artifact: Dict[str, Any]) -> Dict[str, Any]:
    input_data: Dict[str, Any] = {}
    input_numeric_columns = artifact.get("input_numeric_columns", [])
    input_categorical_columns = artifact.get("input_categorical_columns", [])
    input_schema = artifact.get("input_schema", {"numeric": {}, "categorical": {}})

    cols = st.columns(2)

    with st.form("churn_form"):
        # Numeric inputs
        num_cols_per_panel = max(1, (len(input_numeric_columns) + 1) // 2)
        num_left = input_numeric_columns[:num_cols_per_panel]
        num_right = input_numeric_columns[num_cols_per_panel:]

        with cols[0]:
            for c in num_left:
                if c == "SeniorCitizen":
                    input_data[c] = st.selectbox(label=f"{c}", options=[0, 1], index=0)
                else:
                    s = input_schema["numeric"].get(c, {"min": 0.0, "max": 1.0, "step": 0.01})
                    input_data[c] = st.number_input(
                        label=f"{c}",
                        min_value=float(s["min"]),
                        max_value=float(s["max"]),
                        value=float((s["min"] + s["max"]) / 2.0),
                        step=float(s["step"]),
                    )

        with cols[1]:
            for c in num_right:
                if c == "SeniorCitizen":
                    input_data[c] = st.selectbox(label=f"{c}", options=[0, 1], index=0)
                else:
                    s = input_schema["numeric"].get(c, {"min": 0.0, "max": 1.0, "step": 0.01})
                    input_data[c] = st.number_input(
                        label=f"{c}",
                        min_value=float(s["min"]),
                        max_value=float(s["max"]),
                        value=float((s["min"] + s["max"]) / 2.0),
                        step=float(s["step"]),
                    )

        st.divider()

        # Categorical inputs
        cat_cols_per_panel = max(1, (len(input_categorical_columns) + 1) // 2)
        cat_left = input_categorical_columns[:cat_cols_per_panel]
        cat_right = input_categorical_columns[cat_cols_per_panel:]

        with cols[0]:
            for c in cat_left:
                cats = input_schema["categorical"].get(c, {}).get("categories", [])
                if len(cats) <= 25:
                    input_data[c] = st.selectbox(label=f"{c}", options=cats if cats else [""])
                else:
                    selected = st.selectbox(
                        label=f"{c} (top categories)",
                        options=cats[:25] if cats else [""],
                    )
                    custom = st.text_input(label=f"{c} (custom value)", value="")
                    input_data[c] = custom if custom.strip() else selected

        with cols[1]:
            for c in cat_right:
                cats = input_schema["categorical"].get(c, {}).get("categories", [])
                if len(cats) <= 25:
                    input_data[c] = st.selectbox(label=f"{c}", options=cats if cats else [""])
                else:
                    selected = st.selectbox(
                        label=f"{c} (top categories)",
                        options=cats[:25] if cats else [""],
                    )
                    custom = st.text_input(label=f"{c} (custom value)", value="")
                    input_data[c] = custom if custom.strip() else selected

        submitted = st.form_submit_button("Predict churn probability")
        return input_data if submitted else {}


def _render_shap_plot(shap_result: Dict[str, Any]) -> None:
    features = shap_result["top_features"]
    values = shap_result["top_shap_values"]

    # Color by sign.
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in values]  # red/green

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = list(range(len(features)))[::-1]
    ax.barh(y_pos, values[::-1], color=colors[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features[::-1])
    ax.set_title("Top SHAP Feature Contributions (Processed Features)")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP value (impact on churn probability)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def main() -> None:
    st.title("Customer Churn Prediction System")
    st.write("Predict churn probability and interpret the model using SHAP.")

    try:
        artifact = get_artifact()
    except Exception as e:
        st.error(f"Could not load trained model: {e}")
        st.info("Run training first to generate `models/best_churn_model.joblib`.")
        return

    # Model metrics (from validation during training)
    st.sidebar.header("Model Performance (Validation)")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if k in artifact.get("metrics", {}):
            st.sidebar.metric(k, f"{artifact['metrics'][k]:.4f}")

    st.sidebar.caption(f"Decision threshold: {artifact.get('threshold', 0.5):.2f}")

    # Optional visualization images if they exist.
    fi_path = os.path.join("models", "feature_importance.png")
    mc_path = os.path.join("models", "model_comparison.png")
    if os.path.exists(fi_path):
        st.sidebar.subheader("Feature Importance")
        st.sidebar.image(fi_path, use_container_width=True)
    if os.path.exists(mc_path):
        st.sidebar.subheader("Model Comparison")
        st.sidebar.image(mc_path, use_container_width=True)

    shap_enabled = st.checkbox("Show SHAP explanation", value=False)

    input_data = _render_inputs(artifact)
    if input_data:
        try:
            proba, pred, _details = predict_churn_proba(input_data, artifact)
            label_mapping = artifact.get("label_mapping", {0: "No", 1: "Yes"})
            label = label_mapping.get(pred, str(pred))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Churn Probability", f"{proba:.4f}", delta=None)
            with col2:
                st.metric("Prediction", f"{label}", delta=None)

            if shap_enabled:
                with st.spinner("Computing SHAP explanation..."):
                    shap_result = explain_with_shap(input_data, artifact, top_k=15)
                st.subheader("Explanation")
                _render_shap_plot(shap_result)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()

