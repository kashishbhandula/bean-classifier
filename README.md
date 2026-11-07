# Dry Bean Variety Classification (ML Assignment 2)

**Author:** Kashish Bhandula | **BITS ID:** 2026AK05429 | M.Tech (AIML), BITS Pilani WILP | Course: Machine Learning – Assignment 2

## a. Problem Statement
Classify dry beans into one of **7 varieties** (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) from 16 geometric features extracted from images of the grains. Five classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on the same dataset, compared on six metrics, and exposed through a Streamlit app deployed on Streamlit Community Cloud.

## b. Dataset Description  [1 mark]
- **Source:** UCI Machine Learning Repository – *Dry Bean Dataset* (Koklu & Ozkan, 2020).
- **Instances:** 13,611 (min. required 500) | **Features:** 16 numeric (min. required 12) | **Target:** `Class` (7 classes, multi-class).
- **Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1–4.
- **Class distribution:** Dermason 3546, Sira 2636, Seker 2027, Horoz 1928, Cali 1630, Barbunya 1322, Bombay 522 (imbalanced).
- **Split:** 80/20 stratified train/test (`random_state=42`) → 10,888 train / 2,723 test. The test split is saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` inside each scikit-learn `Pipeline` (scaler + classifier saved together as one `.joblib`). Target label-encoded (`model/label_encoder.joblib`).

## c. GitHub Repository Link  [1 mark]
https://github.com/kashishbhandula/bean-classifier

Live app: <PASTE YOUR STREAMLIT URL HERE>

```
bean-classifier/
|-- app.py                 # Streamlit app
|-- train_models.py        # trains & saves all 5 models, writes test_data.csv
|-- requirements.txt
|-- README.md
|-- test_data.csv          # held-out test set (2,723 rows) for upload in the app
|-- data/Dry_Bean_Dataset_full.csv
`-- model/                 # *.joblib pipelines, label_encoder.joblib, metrics.json
```

## d. Models Used  [5 marks]
All models are evaluated on the same held-out test set. Precision, Recall, F1 are **macro-averaged**; AUC is **one-vs-rest macro** (multi-class).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.9041 | 0.9827 | 0.9181 | 0.9147 | 0.9162 | 0.8841 |
| KNN | 0.9137 | 0.9858 | 0.9290 | 0.9245 | 0.9265 | 0.8956 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | **0.9243** | 0.9946 | **0.9372** | **0.9350** | **0.9361** | **0.9085** |

Hyper-parameters: LogisticRegression(max_iter=2000); DecisionTree(max_depth=8); KNN(k=7); GaussianNB(); SVC(kernel='rbf', C=10, probability=True).

### Observations  [3 marks]

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline (acc 0.921, AUC 0.995 – the highest AUC). The shape features are largely linearly separable once standardised; fast, stable and interpretable. Slightly below SVM on MCC because it cannot model the non-linear overlap between Sira/Dermason/Seker. |
| Decision Tree | Weakest of the discriminative models on AUC (0.983) and 2nd-lowest accuracy (0.904). Axis-aligned splits on highly correlated size features (Area, Perimeter, ConvexArea) give hard, coarse probability estimates, and depth-limited trees slightly underfit/overfit the boundary classes. Very interpretable and needs no scaling. |
| KNN | Good (acc 0.914, F1 0.927): local neighbourhoods capture the non-linear structure once features are scaled. Slightly below LR/SVM and the heaviest at inference time (the stored training set makes the model file the largest). |
| Naive Bayes | Lowest accuracy/MCC (0.898 / 0.877) because the Gaussian-independence assumption is violated – many features are strongly correlated (e.g. Area/ConvexArea/EquivDiameter). Still has a high AUC (0.992), so its ranking of classes is good while its hard decisions are poorer calibrated. Extremely fast. |
| Support Vector Machine | Best on Accuracy, Precision, Recall, F1 and MCC. The RBF kernel models non-linear class boundaries while margin maximisation controls overfitting. Most confusion remains between visually similar varieties (Sira/Dermason, Cali/Barbunya). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF)** – best on 5 of 6 metrics (Accuracy 0.9243, MCC 0.9085, F1 0.9361); Logistic Regression is a very close second (marginally higher AUC) and is the better choice if simplicity/speed matter. |

## Streamlit App Features
- Upload test CSV (must contain the 16 features + `Class`) or use the bundled `test_data.csv`
- Model selection dropdown (5 models)
- Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report; side-by-side comparison of all 5 models; predictions download
- Models are loaded from `model/` – **no retraining** happens in the app.

## Run locally
```bash
pip install -r requirements.txt
python train_models.py      # optional: regenerates models + test_data.csv
streamlit run app.py
```
# bean-classifier
