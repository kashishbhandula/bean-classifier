# Dry Bean Variety Classification (ML Assignment 2)

**Author:** Kashish Bhandula | **BITS ID:** 2026AK05429 | M.Tech (AIML), BITS Pilani WILP | Machine Learning - Assignment 2

## a. Problem Statement
Classify dry beans into one of **7 varieties** (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) using 16 shape features measured from images of the beans. I trained five classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) on the same dataset, compared them on six metrics and built a Streamlit app to show the results.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository - Dry Bean Dataset (Koklu & Ozkan, 2020).
- **Instances:** 13,611 | **Features:** 16 numeric | **Target:** `Class` (7 classes, multi-class).
- **Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1 to ShapeFactor4.
- **Class distribution:** Dermason 3546, Sira 2636, Seker 2027, Horoz 1928, Cali 1630, Barbunya 1322, Bombay 522 (Bombay has far fewer samples than the rest).
- **Split:** 80/20 stratified train/test (`random_state=42`) giving 10,888 train / 2,723 test. The test split is saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` inside each scikit-learn `Pipeline` (scaler + classifier saved together as one `.joblib`). The target is label-encoded (`model/label_encoder.joblib`).

## c. GitHub Repository Link
https://github.com/kashishbhandula/bean-classifier

Live app: https://bean-classifiergit-n4pjhpzynsldpisu8qnibs.streamlit.app/

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

## d. Models Used
All models are evaluated on the same test set. Since there are 7 classes, Precision, Recall and F1 are macro-averaged and AUC is one-vs-rest (macro).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.9041 | 0.9827 | 0.9181 | 0.9147 | 0.9162 | 0.8841 |
| KNN | 0.9137 | 0.9858 | 0.9290 | 0.9245 | 0.9265 | 0.8956 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | **0.9243** | 0.9946 | **0.9372** | **0.9350** | **0.9361** | **0.9085** |

Settings used: LogisticRegression(max_iter=2000); DecisionTree(max_depth=8); KNN(k=7); GaussianNB(); SVC(kernel='rbf', C=10, probability=True).

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Did very well for a simple model (accuracy 0.921) and had the highest AUC (0.995). After scaling, the features separate fairly well with a linear boundary. It is fast to train, but it misses a few of the beans that look alike (Sira, Dermason, Seker). |
| Decision Tree | Gave the lowest AUC (0.983) and second lowest accuracy (0.904). Many features (Area, Perimeter, ConvexArea) are highly correlated, so the tree splits are not very clean. Easy to understand but not the best here. |
| KNN | Accuracy 0.914 and F1 0.927, which is good once the features are scaled. A bit lower than Logistic Regression and SVM, and it is slower at prediction time because it keeps all the training data (its model file is the biggest). |
| Naive Bayes | Lowest accuracy (0.898) and MCC (0.877). It assumes the features are independent, which is not true here since many are strongly related (Area, ConvexArea, EquivDiameter). Its AUC is still high (0.992), and it is very fast. |
| Support Vector Machine | Best accuracy (0.924), precision, recall, F1 and MCC. The RBF kernel handles the non-linear boundary between classes well. Most of its mistakes are between Sira and Dermason, and between Cali and Barbunya. |
| Overall, Winner for your dataset? | Support Vector Machine, as it is best on 5 of the 6 metrics (accuracy 0.9243, F1 0.9361, MCC 0.9085). Logistic Regression is very close behind and has slightly higher AUC. |

## Streamlit App Features
- Upload test CSV (must contain the 16 features + `Class`) or use the bundled `test_data.csv`
- Model selection dropdown (5 models)
- Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report; comparison of all 5 models; download of predictions
- Models are loaded from `model/` (the app does not retrain anything).

## Run locally
```bash
pip install -r requirements.txt
python train_models.py      # optional: regenerates models + test_data.csv
streamlit run app.py
```
