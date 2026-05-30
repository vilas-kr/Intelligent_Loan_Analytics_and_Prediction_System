# Intelligent Loan Analytics and Prediction System using Machine Learning

This project builds an end-to-end machine learning system on the HDFC Loan Dataset to solve three real-world banking analytics problems: loan approval prediction, loan default risk analysis, and loan amount prediction.

## Project Objective

The goal of this project is to apply data preprocessing, exploratory data analysis, feature engineering, machine learning, model evaluation, and hyperparameter tuning on a banking dataset containing customer demographics, credit history, employment details, behavioral fields, and loan-related attributes.

## Use Cases Implemented

### 1. Loan Approval Prediction
Build a classification model to predict whether a customer's loan application will be approved or rejected.

**Target column:** `Loan_Status`

**Models used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

**Evaluation metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 2. Loan Default Risk Analysis
Build a classification model to identify customers with a high probability of loan default risk.

**Derived target column:** `High_Risk = 1 if Default_History_Count > 0 else 0`

**Models used:**
- Logistic Regression
- Random Forest Classifier

**Evaluation metrics:**
- Recall
- Precision
- F1 Score
- ROC-AUC
- Confusion Matrix

### 3. Loan Amount Prediction
Build a regression model to predict the eligible loan amount for a customer.

**Target column:** `Loan_Amount`

**Models used:**
- Linear Regression
- Random Forest Regressor

**Evaluation metrics:**
- MAE
- MSE
- RMSE
- R² Score

## Dataset Description

The dataset contains customer demographic information, loan details, financial history, credit behavior, employment information, and sentiment-related attributes.

### Important columns
- `Applicant_Income`
- `Coapplicant_Income`
- `Loan_Amount`
- `Loan_Term_Months`
- `Credit_History`
- `CIBIL_Score`
- `Debt_to_Income_Ratio`
- `Existing_EMIs`
- `Employment_Length_Years`
- `Loan_Status`
- `Customer_Sentiment`
- `Purpose_of_Loan`
- `Occupation`
- `Property_Area`
- `State`
- `City`
- `Asset_Value`
- `Monthly_Expense`
- `Annual_Household_Income`

## Project Workflow

### 1. Data Understanding
- Dataset inspection
- Shape, columns, and data types
- Null value analysis
- Duplicate checking
- Statistical summary

### 2. Exploratory Data Analysis
Visualizations created in the project include:
- Loan approval distribution
- Applicant income distribution
- CIBIL score distribution
- EMI vs Loan Amount
- Loan status by property area
- Loan term distribution
- Top states by applications
- Top occupations
- CIBIL score vs loan status
- Loan status by occupation

### 3. Data Preprocessing
- Missing value handling using `SimpleImputer`
- Categorical encoding using `OneHotEncoder`
- Numeric scaling using `StandardScaler`
- Train-test split
- Pipeline-based preprocessing with `ColumnTransformer`

### 4. Feature Engineering
Custom features created for regression and risk analysis include:
- `Total_Applicant_Income`
- `EMI_to_Income`
- `Expense_to_Income`
- `Loan_per_Asset`

### 5. Model Building
Each use case is implemented with multiple machine learning algorithms and compared using suitable evaluation metrics.

### 6. Hyperparameter Tuning
Hyperparameter tuning is performed using `GridSearchCV` on selected models.

Examples:
- Logistic Regression tuning for approval prediction and default risk analysis
- Random Forest tuning for loan amount prediction

## Project Structure

```bash
hdfc_ml_project/
|
|-- dataset/
|   |-- hdfc_loan_dataset_full_enriched.csv
|
|-- notebooks/
|   |-- eda.ipynb
|   |-- loan_approval_prediction.ipynb
|   |-- default_risk_analysis.ipynb
|   |-- loan_amount_prediction.ipynb
|
|-- src/
|   |-- preprocessing.py
|   |-- feature_engineering.py
|   |-- evaluate.py
|
|-- models/
|   |-- loan_approval_pipeline.pkl
|   |-- risk_model.pkl
|   |-- loan_amount_model.pkl
|
|-- visuals/
|   |-- loan_approval_distribution.png
|   |-- applicat_income_distribution.png
|   |-- cibil_score_distribution.png
|   |-- emi_vs_loan_amount.png
|   |-- loans_by_area.png
|   |-- loan_status_by_area.png
|   |-- loan_term_distribution.png
|   |-- top_10_states_by_loan_applications.png
|   |-- top_10_occupations.png
|   |-- cibil_vs_loan_status.png
|   |-- loan_status_by_occupation.png
|
|-- requirements.txt
|-- README.md
```

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/vilas-kr/Intelligent_Loan_Analytics_and_Prediction_System.git
cd Intelligent_Loan_Analytics_and_Prediction_System
```

### 2. Create Virtual Environment
Windows
```bash
python -m venv venv
venv\Scripts\activate
```
Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required libraries using:
```bash
pip install -r requirements.txt
```
### 4. Run Jupyter notebooks
Open the notebooks folder and run:
- `eda.ipynb`
- `loan_approval_prediction.ipynb`
- `default_risk_analysis.ipynb`
- `loan_amount_prediction.ipynb`

### 5. Train models
Run the respective notebook cells or Python scripts to preprocess the data, train the models, evaluate performance, and save the trained pipeline.

## Key Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

##  Learning Outcomes

This project demonstrates:

- End-to-end Machine Learning workflow
- Classification and Regression
- Feature Engineering
- Hyperparameter Tuning
- Model Evaluation
- Pipeline Design
- Data Visualization
- Production-ready ML practices

## Banking Relevance

This project simulates practical banking workflow such as:
- Automated loan approval systems
- Credit risk profiling
- Default risk identification

## Author
```
Name: Vilas K R
GitHub: https://github.com/vilas-kr
```