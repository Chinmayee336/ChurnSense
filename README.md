# ChurnSense — Customer Churn Prediction

A beginner-friendly machine learning project that predicts whether a telecommunications customer is likely to leave a service.

## Features
- Uses the public IBM Telco Customer Churn sample dataset.
- Cleans and preprocesses numeric and categorical columns.
- Trains and compares Logistic Regression and Random Forest.
- Evaluates models using accuracy, precision, recall, F1-score and ROC-AUC.
- Saves the selected model with Joblib.
- Provides an interactive Streamlit dashboard with customer analytics and a prediction form.

## Project structure
```text
customer_churn_prediction/
├── app.py
├── train_model.py
├── download_data.py
├── requirements.txt
├── README.md
├── data/
└── models/
```

## Run on Windows

Open this folder in VS Code. Open Terminal and run:

### 1. Create a virtual environment
```bash
python -m venv venv
```

### 2. Activate it
```bash
venv\Scripts\activate
```

### 3. Install packages
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset
```bash
python download_data.py
```

### 5. Train and evaluate the models
```bash
python train_model.py
```

This creates the trained model and evaluation files inside `models/`.

### 6. Start the dashboard
```bash
streamlit run app.py
```

The dashboard opens in your browser. If it does not open automatically, visit `http://localhost:8501`.

## Dataset
IBM Telco Customer Churn sample dataset:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The dataset contains 7,043 customer records and a `Churn` target column. It is a sample dataset for learning and demonstration, not a live telecom-company database.

## Important
- Run `train_model.py` before starting the dashboard.
- Model performance shown in the report must be copied from your own training run.
- The model's output is an estimate, not a guarantee that a customer will leave.
