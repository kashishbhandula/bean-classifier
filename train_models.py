# Author: Kashish Bhandula (2026AK05429) - ML Assignment 2, BITS Pilani WILP
"""Train 5 classifiers on the UCI Dry Bean dataset and save them with metrics."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 42
df = pd.read_csv("data/Dry_Bean_Dataset_full.csv")
le = LabelEncoder()
y = le.fit_transform(df["Class"])
X = df.drop(columns=["Class"])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# Test CSV used by the Streamlit app (features + original string target)
test_df = X_te.copy()
test_df["Class"] = le.inverse_transform(y_te)
test_df.to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=10, probability=True, random_state=SEED),
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)
    results[name] = {
        "Accuracy": accuracy_score(y_te, pred),
        "AUC": roc_auc_score(y_te, proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_te, pred, average="macro"),
        "Recall": recall_score(y_te, pred, average="macro"),
        "F1": f1_score(y_te, pred, average="macro"),
        "MCC": matthews_corrcoef(y_te, pred),
    }
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, f"model/{fname}")
    print(name, {k: round(v, 4) for k, v in results[name].items()})

joblib.dump(le, "model/label_encoder.joblib")
json.dump(results, open("model/metrics.json", "w"), indent=2)
print(len(X_tr), len(X_te))
