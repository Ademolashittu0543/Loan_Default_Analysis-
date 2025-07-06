# 🏦 Loan Default Prediction with Machine Learning

> A complete end-to-end machine learning pipeline to predict the likelihood of loan default using real-world tabular data.

---

## 📌 Project Overview

This project focuses on predicting whether a loan applicant will default using historical loan data. It covers the full machine learning workflow — from **data preprocessing** to **model selection**, **evaluation**, and **interpretation**.

---

## 📁 Dataset

- **Source:** Provided as `Loan_Default.csv`
- **Size:** 148,000+ records
- **Target Variable:** `Status` (0 = No Default, 1 = Default)
- **Features include:**
  - Loan amount, term, rate of interest
  - Credit score, LTV ratio, income
  - Gender, loan type, approval status
  - Co-applicant credit type, property value, etc.

---

## ⚙️ Workflow Summary

### 1. 🧹 Data Cleaning
- Handled missing values using `SimpleImputer`
- Removed duplicates
- Performed statistical profiling (mean, skewness, kurtosis)
- Created additional features (e.g., `age_midpoint`)

### 2. 📊 Exploratory Data Analysis (EDA)
- Univariate & bivariate visualizations with `matplotlib`, `seaborn`, and `plotly`
- Detected right-skewed distributions and outliers
- Observed that ~25% of loans defaulted (imbalanced dataset)

### 3. 🧠 Feature Selection
- Used **Recursive Feature Elimination (RFE)** for wrapper-based selection
- Compared model performance with **all features vs. selected features**

### 4. 🤖 Model Training
- Built pipelines with:
  - `LogisticRegression`
  - `RandomForestClassifier`
  - `XGBoostClassifier`
  - `KNeighborsClassifier`
- Handled class imbalance with **SMOTE**
- Applied **GridSearchCV** for hyperparameter tuning

### 5. 📈 Evaluation
- Metrics used:
  - F1-score
  - ROC-AUC
  - Classification Report
- Plotted **learning curves** to detect overfitting/underfitting
- ROC and Confusion Matrix visualization for all models

---

## 🚀 Results Summary

| Model               | Test F1 Score | ROC-AUC | Conclusion        |
|---------------------|---------------|---------|-------------------|
| Logistic Regression | 0.66          | 0.87    | ✅ Generalizes Well |
| Random Forest       | 1.00          | 1.00    | ✅ Excellent Performance |
| XGBoost             | 1.00          | 1.00    | ✅ Excellent Performance |
| KNN                 | 0.78          | 0.92    | 🟡 Some Overfitting |

---

## 🧪 Tools & Libraries

- Python (Pandas, NumPy)
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- Matplotlib / Seaborn / Plotly
- Jupyter Notebook

---

## 🙋‍♂️ Author

**Ademola Shittu**  
📧 ademolashittu0543@gmail.com  

---

## ⭐️ Show your support

If you liked this project:
- ⭐ Star this repo
- 🔁 Share it with friends
- 🍕 Let’s connect and collaborate!
