from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from download_data import main as download_dataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]

def load_and_prepare_data():
    if not DATA_PATH.exists():
        download_dataset()

    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})

    X = df.drop(columns=DROP_COLUMNS + [TARGET])
    y = df[TARGET].astype(int)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return df, X, y, preprocessor

def main():
    df, X, y, preprocessor = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    }

    results = []
    fitted_models = {}

    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        scores = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, probabilities)
        }
        results.append(scores)
        fitted_models[name] = pipeline
        print(f"\n{name}")
        print(classification_report(y_test, predictions, target_names=["Stay", "Churn"]))

    results_df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False)
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)

    joblib.dump(best_model, MODEL_DIR / "churn_model.joblib")
    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    cm = confusion_matrix(y_test, best_predictions).tolist()
    metadata = {
        "best_model": best_name,
        "test_rows": int(len(y_test)),
        "dataset_rows": int(len(df)),
        "features": X.columns.tolist(),
        "confusion_matrix": cm,
        "metrics": {
            key: float(value)
            for key, value in results_df.iloc[0].to_dict().items()
            if key != "Model"
        }
    }
    with open(MODEL_DIR / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model and evaluation files in: {MODEL_DIR}")

if __name__ == "__main__":
    main()
