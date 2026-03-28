## Customer-churn-analysis
A supervised machine learning project that predicts whether a bank customer will churn (leave), using Logistic Regression and XGBoost — with a focus on handling class imbalance.

## Problem Statement
Customer churn is a critical challenge for banks. Retaining an existing customer is far more cost-effective than acquiring a new one. This project builds a binary classification model to predict which customers are likely to churn, enabling a retention team to intervene proactively.

## Key Features Used
| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Country (France, Germany, Spain) |
| `Gender` | Male / Female |
| `Age` | Customer's age |
| `Tenure` | Number of years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is currently active |
| `EstimatedSalary` | Estimated annual salary |
 
---
## Project Workflow
 
```
Data Loading → EDA → Preprocessing → Model Training → Evaluation → Comparison
```
 
1. **Exploratory Data Analysis** — checked distributions, null values, and class imbalance
2. **Feature Engineering** — dropped leaky/irrelevant columns, label-encoded `Gender`, one-hot encoded `Geography` and `Card Type`
3. **Scaling** — applied `StandardScaler` (fit on train only, applied to test)
4. **Modelling** — trained Logistic Regression (baseline) and XGBoost (tuned for imbalance)
5. **Evaluation** — assessed accuracy, precision, recall, F1-score, and confusion matrix
 
---
 
## Results
 
| Metric | Logistic Regression | XGBoost |
|---|---|---|
| Overall Accuracy | 81.2% | 80.4% |
| Churn Precision | 61% | 51% |
| **Churn Recall** | **21%** | **75%** |
| Churn F1-Score | 0.31 | 0.61 |
 
### Key Finding
 
> Logistic Regression achieved higher overall accuracy but almost completely failed to identify actual churners (21% recall). XGBoost — tuned with `scale_pos_weight=4` to address class imbalance — raised churn recall to **75%**, making it far more useful for a real-world retention campaign.
 
---
 
## Tech Stack
 
| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualisation |
| `scikit-learn` | Preprocessing, Logistic Regression, evaluation metrics |
| `xgboost` | Gradient boosting classifier |
| `kagglehub` | Dataset download |
 
---

## Future Improvements
 
* Hyperparameter tuning with `GridSearchCV` or `Optuna`
* Try additional models: Random Forest, LightGBM, CatBoost
* Build a simple Streamlit app for interactive predictions
 
---
