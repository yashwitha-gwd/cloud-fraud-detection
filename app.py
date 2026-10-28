
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("fraud_model_rf.joblib")

model = load_model()

st.title("🛡️ Credit Card Fraud Detection System")
st.write(
    "Enter transaction details to assess whether a transaction "
    "may be fraudulent."
)

st.info(
    "This model was trained using the public Credit Card Fraud "
    "Detection dataset. Predictions are estimates, not guarantees."
)

st.subheader("Transaction Details")

# Features V1 to V28
transaction = {}

col1, col2 = st.columns(2)

for i in range(1, 29):
    key = f"V{i}"

    with col1 if i <= 14 else col2:
        transaction[key] = st.number_input(
            key,
            value=0.0,
            format="%.6f",
            help=f"Enter the {key} feature value."
        )

col3, col4 = st.columns(2)

with col3:
    transaction["Time"] = st.number_input(
        "Time (seconds)",
        min_value=0.0,
        value=0.0
    )

with col4:
    transaction["Amount"] = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

if st.button("🔍 Analyze Transaction", type="primary"):
    features = pd.DataFrame(
        [[transaction[column] for column in
          ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]]],
        columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    )

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("Analysis Result")

    if prediction == 1:
        st.error("🚨 Potential Fraud Detected")
    else:
        st.success("✅ Classified as Legitimate")

    st.metric(
        "Estimated Fraud Probability",
        f"{probability:.2%}"
    )

    st.caption(
        "The result depends on the model and the input values. "
        "A low estimated probability does not guarantee that a "
        "transaction is safe."
    )

