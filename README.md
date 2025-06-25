# Predicting-Heart-Disease-Risk-Using-Supervised-Machine-Learning
Predicting the presence of heart disease based on clinical and demographic health data


# Heart Disease Risk Prediction using Supervised Machine Learning

This project applies supervised machine learning techniques to predict the presence of heart disease based on structured clinical and demographic data. We use a well-known dataset from the UCI Machine Learning Repository, accessed via Kaggle, to develop, train, and evaluate several models for binary classification.

## Dataset

- **Name:** UCI Heart Disease Dataset (Cleveland subset)
- **Source:** [Kaggle - redwankarimsony](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Rows:** 920 patient records
- **Features:** 14 clinical and demographic variables
- **Target:** `target` (0 = No Heart Disease, 1 = Heart Disease)

### Selected Features

- Demographic: age, sex
- Clinical: chest pain type, resting BP, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, slope, number of vessels, thalassemia

### Notes
- Missing values handled using median/mode imputation.
- Target variable converted to binary from original multi-class `num` column.

---

## Problem Statement

Can we build an accurate, interpretable, and lightweight machine learning model that predicts the presence of heart disease from patient data?

This binary classification task is critical for medical screening and early diagnosis, helping reduce risk through timely intervention.

---

## Methods and Models Used

### Preprocessing
- Imputation of missing values
- One-hot encoding of categorical variables
- Standardization of numerical features
- Train-test split with stratification

### Models Trained
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Tuned Random Forest (via `RandomizedSearchCV`)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Reports
- Visual bar chart comparison

---

## Results Summary

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.842    | 0.841     | 0.882  | 0.861    |
| Decision Tree       | 0.799    | 0.810     | 0.833  | 0.821    |
| Random Forest       | 0.859    | 0.845     | 0.912  | 0.877    |
| Tuned Random Forest | 0.840    | 0.850     | 0.830  | 0.840    |

- **Best model:** Untuned Random Forest (F1 Score = 0.877)
- **Tuned model:** More balanced metrics, but slightly lower overall performance

---

## Conclusions

- **Random Forest** outperformed other models in accuracy and recall, making it ideal for detecting heart disease.
- **Tuned Random Forest** offered better balance between precision and recall, reducing false positives and false negatives—important in clinical applications.
- **Logistic Regression** provided strong baseline performance and interpretability.
- **Decision Tree** was less effective but easy to visualize.

---

## Future Improvements

- Add more data from other heart disease datasets to improve generalizability.
- Test more advanced models (e.g., XGBoost, LightGBM, or neural networks).
- Explore SHAP or LIME for deeper model interpretability.
- Improve hyperparameter tuning with grid search or Bayesian optimization.

---

## How to Run This Project

### 1. Clone this repository

```bash
git clone https://github.com/nushin01/heart-disease-risk-prediction.git
cd heart-disease-risk-prediction



Make sure you have Python ≥ 3.8 installed. Then run:

pip install -r requirements.txt


Run the Jupyter Notebook : Supervised Learning Final Project.ipynb

## Project Structure

heart-disease-risk-prediction/
│
├── data/
│   └── heart_disease_uci.csv
├── Heart_Disease_Classifier.ipynb
├── requirements.txt
└── README.md

### License
This project is for academic and educational purposes only. The dataset is available under the public domain via UCI and Kaggle.


Author
Nushin Anwar
Introduction to Machine Learning: Supervised Learning Final Project
CU Boulder
