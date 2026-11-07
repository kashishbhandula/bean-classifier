# Author: Kashish Bhandula (2026AK05429) - ML Assignment 2, BITS Pilani WILP
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "Class"

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(BASE, "model", MODEL_FILES[name]))


@st.cache_resource
def load_encoder():
    return joblib.load(os.path.join(BASE, "model", "label_encoder.joblib"))


st.title("🫘 Dry Bean Variety Classification")
st.caption("Pre-trained models (no retraining) evaluated on an uploaded test CSV. "
           "Dataset: UCI Dry Bean (7 varieties, 16 shape features).")

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Select model", list(MODEL_FILES))
    uploaded = st.file_uploader("Upload test CSV", type="csv",
                                help="Must contain the 16 feature columns and a 'Class' column.")
    use_default = st.checkbox("Use bundled test_data.csv", value=uploaded is None)

if uploaded is not None:
    data = pd.read_csv(uploaded)
elif use_default:
    data = pd.read_csv(os.path.join(BASE, "test_data.csv"))
else:
    st.info("Upload a test CSV (or tick the bundled test data) to begin.")
    st.stop()

pipe = load_model(model_name)
le = load_encoder()
features = list(pipe.feature_names_in_)

missing = [c for c in features + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

unknown = set(data[TARGET].unique()) - set(le.classes_)
if unknown:
    st.error(f"Unknown class labels in target column: {sorted(unknown)}")
    st.stop()

X = data[features]
y_true = le.transform(data[TARGET])
y_pred = pipe.predict(X)
y_proba = pipe.predict_proba(X)

st.subheader(f"Data preview ({len(data):,} rows)")
st.dataframe(data.head(10), width="stretch")

# ---- metrics ----
try:
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                        labels=range(len(le.classes_)))
except ValueError:
    auc = float("nan")  # happens if the uploaded subset lacks some classes
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": auc,
    "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

st.subheader(f"Evaluation metrics — {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")
st.caption("Precision / Recall / F1 / AUC are macro-averaged (AUC is one-vs-rest) for the 7-class problem.")

# ---- comparison of all models on this data ----
with st.expander("Compare all 5 models on this data", expanded=False):
    rows = {}
    for n in MODEL_FILES:
        p = load_model(n)
        yp, pr = p.predict(X), p.predict_proba(X)
        try:
            a = roc_auc_score(y_true, pr, multi_class="ovr", average="macro", labels=range(len(le.classes_)))
        except ValueError:
            a = float("nan")
        rows[n] = {
            "Accuracy": accuracy_score(y_true, yp), "AUC": a,
            "Precision": precision_score(y_true, yp, average="macro", zero_division=0),
            "Recall": recall_score(y_true, yp, average="macro", zero_division=0),
            "F1": f1_score(y_true, yp, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, yp),
        }
    st.dataframe(pd.DataFrame(rows).T.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"),
                 width="stretch")

# ---- confusion matrix + report ----
left, right = st.columns([1, 1])
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(le.classes_)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y_true, y_pred, labels=range(len(le.classes_)),
                                target_names=le.classes_, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.style.format("{:.3f}"), width="stretch")

st.download_button("Download predictions (CSV)",
                   data.assign(Predicted=le.inverse_transform(y_pred)).to_csv(index=False),
                   file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv")
