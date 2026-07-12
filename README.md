# Loan_Default_Risk_Prediction
# Loan Default Risk Prediction

Predicting whether a loan applicant will face payment difficulties, using the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset from Kaggle.

## Problem Statement

Given an applicant's demographic, income, and loan details, predict the likelihood that they will have payment difficulties (`TARGET = 1`) versus repay normally (`TARGET = 0`).

This is a binary classification problem on real-world, messy financial data — with a significant class imbalance (~92% no-difficulty vs ~8% payment-difficulty cases).

## Dataset

- **Source:** `application_train.csv` and `application_test.csv` from the Home Credit Default Risk competition
- Only the main application table was used (not the additional bureau/previous-application/installment tables), to keep the project focused on core feature engineering and modeling rather than an exhaustive multi-table join

## Approach & Reasoning

### 1. Data Type Separation
Columns were split into integer, float (numerical), and categorical (object) types. This made it easier to apply the right cleaning strategy to each group instead of a one-size-fits-all approach.

### 2. Handling Missing Values

**Numerical columns:**
- Columns with more than 50% missing values were flagged for removal
- Before dropping, each flagged column's correlation with `TARGET` was checked
- `EXT_SOURCE_1` was the one exception — despite having high missing %, it showed a notably strong correlation with `TARGET` compared to other columns, so it was kept and imputed instead of dropped
- **Why:** blindly dropping by missing % alone risks throwing away a genuinely predictive feature. Checking correlation first avoids that mistake.
- Remaining numerical missing values were filled with the **median** (robust to outliers/skew, which is common in financial data like income and credit amount)

**Categorical columns:**
- Missing values were filled with a new category, `'Unknown'`, rather than the mode
- **Why:** in this dataset, missing categorical values often reflect a real category (e.g., "not provided") rather than random noise. Using a placeholder like `'Unknown'` preserves that information instead of artificially forcing rows into the most common category.

### 3. Removing Low-Variance Columns
`VarianceThreshold` was used to identify numerical columns with very little variation across rows.

**Why:** a column where nearly every value is the same carries little to no predictive signal, regardless of its scale, and adds unnecessary noise/dimensionality to the model. KDE plots were used to visually confirm these low-variance columns before removing them.

### 4. Encoding Categorical Variables
- **Binary categorical columns** (exactly 2 unique values) → encoded with `LabelEncoder`
- **Multi-category columns** (more than 2 unique values) → encoded with one-hot encoding (`pd.get_dummies`, `drop_first=True`)

**Why:** Label encoding is safe only for binary columns since it doesn't impose a false ordinal relationship. For columns with more categories, one-hot encoding avoids the model misinterpreting categories as having a numeric order (e.g., treating "Married" as greater than "Single").

### 5. Handling Class Imbalance
The dataset is heavily imbalanced (~92% vs ~8%). **SMOTE (Synthetic Minority Oversampling Technique)** was applied on the training set only, to synthetically balance the two classes before model training.

**Why:** without addressing imbalance, a model can achieve high accuracy simply by predicting the majority class every time, while completely failing to catch actual payment-difficulty cases — which are the ones that matter most for this problem.

### 6. Model: Random Forest
A `RandomForestClassifier` was trained on the SMOTE-balanced training data.

**Why Random Forest:**
- Handles a mix of numerical and encoded categorical features well
- Naturally scale-invariant (no feature scaling required)
- Provides feature importance out of the box, useful for understanding which factors drive predictions

### 7. Evaluation Metrics
Instead of relying on accuracy, the following were used:
- **Confusion Matrix** — to see actual counts of correct/incorrect predictions per class
- **ROC-AUC Score** — to measure how well the model separates the two classes across all thresholds
- **Classification Report** (precision, recall, F1) — to check performance specifically on the minority (payment-difficulty) class

**Why not accuracy:** with ~92% of the data belonging to one class, a model predicting "no difficulty" for every applicant would already score ~92% accuracy while being completely useless. The metrics above give a much more honest picture of performance on the class that actually matters.

### 8. Feature Importance & Selection
After training, feature importances from the Random Forest were extracted and sorted to identify which features contributed the most (and least) to predictions. Low-importance features (below a small threshold) were identified as candidates for removal in a future iteration, to simplify the model without much loss in performance.

### 9. Hyperparameter Tuning
`RandomizedSearchCV` was used to search over key Random Forest parameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) using ROC-AUC as the scoring metric, with 3-fold cross-validation.

**Why RandomizedSearchCV over GridSearchCV:** with several hyperparameters and multiple values each, an exhaustive grid search becomes computationally expensive. Randomized search samples a fixed number of combinations, giving a good balance between search quality and training time.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (modeling, encoding, feature selection, evaluation)
- imbalanced-learn (SMOTE)
- Matplotlib, Seaborn (visualization)

## Possible Next Steps
- Try gradient boosting models (XGBoost/LightGBM), which typically perform better on tabular, imbalanced data
- Add SHAP for model explainability — understanding *why* a specific applicant is flagged as high risk
- Convert the default 0.5 classification threshold into a business-cost-based threshold (weighing the cost of missed defaults vs rejected good applicants)
- Incorporate the additional bureau/previous-application tables for richer features
