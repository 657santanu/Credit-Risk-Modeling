# 💳 Credit-Risk-Modeling

An end-to-end **Machine Learning Classification Project** that predicts credit risk (**good vs. bad**) for loan applicants using the **German Credit Data** dataset.

The project covers data cleaning, exploratory data analysis, feature engineering, categorical encoding, model training, hyperparameter tuning, and model serialization using Python and Scikit-learn.

---

## 📌 Project Overview

The objective of this project is to transform raw applicant data into a machine learning model capable of classifying loan applicants as **good** or **bad** credit risks.

### Key Technologies

* **Python** – Data processing and machine learning
* **Pandas & NumPy** – Data manipulation and numerical operations
* **Matplotlib & Seaborn** – Data visualization and EDA
* **Scikit-learn** – Preprocessing, model training, tuning, and evaluation
* **XGBoost** – Gradient-boosted classification
* **Joblib** – Model and encoder serialization
* **Jupyter Notebook** – Development and experimentation

---

## 📂 Dataset

The project uses the following CSV dataset:

```text
german_credit_data.csv
```

### Dataset Summary

| Stage            |  Rows | Columns |
| ---------------- | ----: | ------: |
| Original dataset | 1,000 |      11 |
| After cleaning   |   522 |      10 |

### Dataset Features

| Column             | Description                                |
| ------------------ | ------------------------------------------ |
| `Age`              | Applicant's age                            |
| `Sex`              | Applicant's gender                         |
| `Job`              | Job category represented by a numeric code |
| `Housing`          | Housing type: own, rent, or free           |
| `Saving accounts`  | Savings account level                      |
| `Checking account` | Checking account level                     |
| `Credit amount`    | Loan/credit amount requested               |
| `Duration`         | Loan duration in months                    |
| `Purpose`          | Purpose of the loan                        |
| `Risk`             | Target variable: good or bad credit risk   |

---

# 🛠 Tools & Technologies

| Technology       | Purpose                                         |
| ---------------- | ----------------------------------------------- |
| Python           | Core data analysis and modeling                 |
| Pandas           | Data manipulation                               |
| NumPy            | Numerical operations                            |
| Matplotlib       | Data visualization                              |
| Seaborn          | Statistical visualization                       |
| Scikit-learn     | Preprocessing, modeling, tuning, and evaluation |
| XGBoost          | Gradient-boosted classification                 |
| Joblib           | Model and encoder serialization                 |
| Jupyter Notebook | Development environment                         |

---

# 🔄 Project Workflow

## 1. Data Loading

The dataset was loaded using Pandas and initially inspected for its structure, dimensions, and data quality.

```python
import pandas as pd

df = pd.read_csv("german_credit_data.csv")

df.head()
df.shape
```

---

## 2. Data Cleaning

The following data-cleaning steps were performed:

* Analyzed missing values
* Checked for duplicate records
* Removed the redundant `Unnamed: 0` index column
* Removed rows containing missing values
* Reset the DataFrame index

### Missing Values

The dataset contained missing values in:

* `Saving accounts` – 183 missing values
* `Checking account` – 394 missing values

No duplicate records were found.

Rows containing missing values were removed, reducing the dataset from **1,000 rows to 522 usable records**.

```python
df.isnull().sum()

df = df.dropna().reset_index(drop=True)

df.drop(columns="Unnamed: 0", inplace=True)
```

> **Note:** Dropping rows with missing values resulted in a substantial reduction in dataset size. A future version could investigate imputation techniques to retain more observations.

---

# 📊 3. Exploratory Data Analysis

Extensive EDA was performed to understand applicant characteristics, credit behavior, feature relationships, and differences between good and bad credit risks.

### Numerical Feature Analysis

Analyzed:

* Age
* Credit amount
* Duration

Visualizations included:

* Histograms
* Boxplots
* Distribution plots
* Scatterplots

Outliers were also inspected, including unusually long loan durations such as `Duration > 70`.

### Categorical Feature Analysis

Analyzed the distribution of:

* Sex
* Job
* Housing
* Saving accounts
* Checking account
* Purpose

### Correlation Analysis

A correlation heatmap was created to examine relationships among numerical variables.

```python
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
```

### Additional Analysis

The project also explored:

* Credit amount by Job
* Credit amount by Sex
* Housing × Purpose relationships
* Age vs. Credit amount
* Saving accounts vs. Credit amount
* Feature distributions segmented by Risk
* Credit amount and loan duration patterns

Example visualization:

```python
sns.scatterplot(
    data=df,
    x="Age",
    y="Credit amount",
    hue="Sex",
    size="Duration"
)
```

---

# ⚙️ 4. Feature Engineering & Encoding

The following features were selected for model training:

```text
Age
Sex
Job
Housing
Saving accounts
Checking account
Credit amount
Duration
```

### Target Variable

```text
Risk
```

Categorical variables were converted into numerical representations using **Label Encoding**.

Each encoder was saved using Joblib so that the same encoding mappings can be reused during inference.

```python
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    joblib.dump(le, f"{col}_encoder.pkl")
```

Saved encoders include:

```text
Sex_encoder.pkl
Job_encoder.pkl
Housing_encoder.pkl
Saving_accounts_encoder.pkl
Checking_account_encoder.pkl
target_encoder.pkl
```

---

# 🧪 5. Train/Test Split

The dataset was divided into training and testing sets using an **80/20 split**.

A **stratified split** was used to preserve the class distribution of the target variable across the training and testing datasets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=1
)
```

---

# 🤖 6. Model Training & Hyperparameter Tuning

Four classification algorithms were trained and optimized using **GridSearchCV** with **5-fold cross-validation**.

The models evaluated were:

1. Decision Tree
2. Random Forest
3. Extra Trees
4. XGBoost

Class imbalance was addressed using:

* `class_weight="balanced"` for applicable Scikit-learn models
* `scale_pos_weight` for XGBoost

### GridSearchCV

A reusable training function was created to perform hyperparameter tuning and evaluate each model on the test set.

```python
def train_model(
    model,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test
):
    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    return best_model, acc, grid.best_params_
```

---

# 📈 7. Model Results

| Model         | Test Accuracy |
| ------------- | ------------: |
| Decision Tree |         58.1% |
| Random Forest |         61.9% |
| Extra Trees   |         64.8% |
| **XGBoost**   |     **67.6%** |

### 🏆 Best Performing Model

**XGBoost achieved the highest test accuracy of 67.6%.**

Based on the evaluated models, XGBoost was the strongest-performing classifier for this dataset.

---

# ⚠️ Model Serialization Note

The current notebook serializes the **Extra Trees model**:

```text
extra_trees_credit_model.pkl
```

However, **XGBoost achieved the highest test accuracy (67.6%)**.

Therefore, the model-selection and serialization logic should be revisited so that the best-performing model is automatically selected and saved.

A better approach would be to compare all trained models and serialize the model with the highest validated performance.

---

# 📁 Project Structure

```text
German-Credit-Risk-Classification/
│
├── data/
│   └── german_credit_data.csv
│
├── notebooks/
│   └── analysis_model.ipynb
│
├── models/
│   ├── extra_trees_credit_model.pkl
│   ├── Sex_encoder.pkl
│   ├── Job_encoder.pkl
│   ├── Housing_encoder.pkl
│   ├── Saving_accounts_encoder.pkl
│   ├── Checking_account_encoder.pkl
│   └── target_encoder.pkl
│
├── images/
│   ├── correlation_heatmap.png
│   └── eda_distributions.png
│
├── requirements.txt
│
└── README.md
```

---

# 🏗️ Project Architecture

```text
Raw Dataset
(german_credit_data.csv)
        │
        ▼
Data Cleaning
(Missing Values + Duplicates)
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Selection
        │
        ▼
Categorical Encoding
        │
        ▼
Stratified Train/Test Split
        │
        ▼
Model Training
        │
        ▼
GridSearchCV Hyperparameter Tuning
        │
        ├── Decision Tree
        ├── Random Forest
        ├── Extra Trees
        └── XGBoost
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
        │
        ▼
Serialized Model + Encoders
```

---

# 🚀 Future Improvements

Several improvements can be made to make the project more robust and production-ready.

### 1. Improve Missing Value Handling

Instead of dropping all rows with missing values, investigate:

* Median/mode imputation
* Categorical imputation
* Model-based imputation

This could help retain more of the original 1,000 observations.

### 2. Improve Model Evaluation

Accuracy alone may not provide a complete picture of credit-risk performance.

Future evaluation should include:

* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC
* Precision-Recall AUC

### 3. Address Class Imbalance

Experiment with:

* SMOTE
* Random oversampling
* Random undersampling
* Class-weight optimization

### 4. Feature Importance

Analyze which applicant characteristics contribute most to credit-risk predictions using:

* Feature importance
* Permutation importance
* SHAP

### 5. Automated Best-Model Selection

Instead of manually selecting the serialized model, automatically save the model with the best cross-validation or validation performance.

### 6. Build an Inference Pipeline

Create a reusable prediction pipeline or API that accepts applicant information and returns the predicted credit risk.

Possible technologies:

* FastAPI
* Flask
* Streamlit

### 7. Model Explainability

Implement SHAP or similar explainability techniques to understand why a particular applicant was classified as good or bad credit risk.

---

# 📌 Key Results

* Cleaned and validated the German Credit dataset.
* Reduced the dataset from **1,000 to 522 usable records** after removing missing values.
* Conducted extensive exploratory data analysis across numerical and categorical variables.
* Investigated relationships between applicant characteristics, loan amounts, duration, and credit risk.
* Applied categorical feature encoding using LabelEncoder.
* Used an **80/20 stratified train/test split**.
* Trained and tuned **four classification models** using GridSearchCV and 5-fold cross-validation.
* **XGBoost achieved the highest test accuracy of 67.6%.**
* Serialized model encoders for reuse during inference.
* Identified an opportunity to improve model serialization so that the highest-performing model is automatically saved.

---

# 💡 Conclusion

This project demonstrates an end-to-end machine learning workflow for **credit risk classification**, starting from raw applicant data and progressing through data cleaning, EDA, feature engineering, model training, hyperparameter tuning, evaluation, and model serialization.

Among the four evaluated classifiers, **XGBoost delivered the best test accuracy at 67.6%**.

The project can be further enhanced through better missing-value treatment, more comprehensive evaluation metrics, explainable AI techniques, automated model selection, and deployment through an inference API or interactive application.
