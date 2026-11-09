from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
RESULTS_PATH = ROOT / "models" / "training_results.json"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"

st.set_page_config(
    page_title="ChurnSense | Customer Analytics",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #475569 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>ChurnSense</h1>
  <p>Customer churn prediction and retention insights powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists() or not RESULTS_PATH.exists():
    st.error("The trained model is not available yet.")
    st.markdown("Open the project folder in a terminal and run:")
    st.code("python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

if DATA_PATH.exists():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
else:
    data = None

tab_overview, tab_predict, tab_model = st.tabs(
    ["📊 Overview", "🔎 Predict Churn", "🧠 Model Performance"]
)

with tab_overview:
    st.subheader("Customer overview")
    if data is None:
        st.warning("Dataset file is missing. Run python download_data.py.")
    else:
        total_customers = len(data)
        churn_count = int((data["Churn"] == "Yes").sum())
        stay_count = total_customers - churn_count
        churn_rate = churn_count / total_customers * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total customers", f"{total_customers:,}")
        c2.metric("Customers churned", f"{churn_count:,}")
        c3.metric("Customers retained", f"{stay_count:,}")
        c4.metric("Churn rate", f"{churn_rate:.1f}%")

        left, right = st.columns(2)
        with left:
            st.markdown("**Customer churn distribution**")
            counts = data["Churn"].value_counts().reindex(["No", "Yes"])
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.pie(counts, labels=["Stayed", "Churned"], autopct="%1.1f%%",
                   startangle=90, wedgeprops={"width": 0.42})
            ax.set_title("Churn vs retained customers")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown("**Churn by contract type**")
            contract_churn = pd.crosstab(
                data["Contract"], data["Churn"], normalize="index"
            ) * 100
            fig, ax = plt.subplots(figsize=(5, 3.5))
            contract_churn.reindex(columns=["No", "Yes"]).plot(
                kind="bar", stacked=True, ax=ax
            )
            ax.set_ylabel("Customers (%)")
            ax.set_xlabel("")
            ax.legend(["Stayed", "Churned"], title="Outcome")
            ax.set_title("Churn percentage by contract")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("**Dataset preview**")
        st.dataframe(data.drop(columns=["customerID"]).head(10), use_container_width=True)

with tab_predict:
    st.subheader("Predict customer churn")
    st.write("Enter customer details below to estimate the likelihood of churn.")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with col2:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        with col3:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0)
            total = st.number_input("Total charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

        submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0][1])

        st.divider()
        if prediction == 1:
            st.error("Prediction: Customer is likely to churn")
        else:
            st.success("Prediction: Customer is likely to stay")
        st.metric("Estimated churn probability", f"{probability * 100:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("This is a model estimate based on the sample dataset, not a guarantee of a customer's future decision.")

with tab_model:
    st.subheader("Model evaluation")
    st.write(f"Selected model: **{results['best_model']}**")
    metrics = results["metrics"]
    metric_cols = st.columns(5)
    for col, metric_name in zip(
        metric_cols, ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    ):
        col.metric(metric_name, f"{metrics[metric_name]:.3f}")

    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH)
        st.markdown("**Model comparison on the test dataset**")
        st.dataframe(
            comparison.style.format({
                "Accuracy": "{:.3f}", "Precision": "{:.3f}",
                "Recall": "{:.3f}", "F1 Score": "{:.3f}", "ROC AUC": "{:.3f}"
            }),
            use_container_width=True
        )

    st.markdown("**Confusion matrix for the selected model**")
    cm = results["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Stayed", "Churned"],
                yticklabels=["Stayed", "Churned"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

st.markdown("---")
st.markdown(
    '<p class="small-note">Academic demonstration project • Dataset: IBM Telco Customer Churn sample • '
    'Predictions should support analysis, not replace business decisions.</p>',
    unsafe_allow_html=True
)
