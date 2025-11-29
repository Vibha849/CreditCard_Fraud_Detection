import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTENC


# ------------------------------------
# Streamlit Page Setup + CSS
# ------------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.markdown("""
<style>
.big-title {
    font-size:42px;
    font-weight:700;
    text-align:center;
    color:#2C3E50;
}
.section-title {
    font-size:28px;
    font-weight:700;
    margin-top:30px;
    color:#34495E;
}
.card {
    padding:20px;
    border-radius:12px;
    background-color:#F8F9F9;
    border:1px solid #EAECEE;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------
# Title
# ------------------------------------
st.markdown("<div class='big-title'>💳 Credit Card Fraud Detection Dashboard</div>", unsafe_allow_html=True)
st.write("Analyze, balance, train and predict credit card fraud in one place.")


# ------------------------------------
# File Upload
# ------------------------------------
uploaded_file = st.file_uploader("📥 Upload Credit Fraud Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ===============================
    # 1. DATASET PREVIEW
    # ===============================
    st.markdown("<div class='section-title'>📌 Dataset Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    # Remove non-feature columns
    X = df.drop(columns=["IsFraud"])
    y = df["IsFraud"]

    X = X.drop(columns=["TransactionID", "CustomerID", "TransactionTime"])

    numeric_features = ["TransactionAmount", "AccountAgeDays", "RiskScore", "Hour"]
    categorical_features = ["MerchantCategory", "TransactionMode", "Location", "DayOfWeek"]

    # ===============================
    # 2. EDA
    # ===============================
    st.markdown("<div class='section-title'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud Count")
        fig1 = plt.figure(figsize=(5, 4))
        sns.countplot(data=df, x="IsFraud")
        st.pyplot(fig1)

    with col2:
        st.subheader("Transaction Amount Distribution")
        fig2 = plt.figure(figsize=(5, 4))
        sns.histplot(df["TransactionAmount"], bins=40)
        st.pyplot(fig2)

    # ===============================
    # 3. SMOTENC FIX — No Errors
    # ===============================
    st.markdown("<div class='section-title'>⚙️ Data Preprocessing (with SMOTENC)</div>", unsafe_allow_html=True)

    # Get categorical indexes
    categorical_idx = [X.columns.get_loc(col) for col in categorical_features]

    smote = SMOTENC(
        categorical_features=categorical_idx,
        random_state=42
    )

    X_resampled, y_resampled = smote.fit_resample(X, y)

    st.success("SMOTENC Applied Successfully — No String Errors!")

    st.write("Class distribution after SMOTENC:")
    st.dataframe(
        pd.Series(y_resampled, name="IsFraud").value_counts().reset_index(),
        use_container_width=True
    )

    # ===============================
    # 4. ENCODE AFTER SMOTE
    # ===============================
    pre_encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    X_encoded = pre_encoder.fit_transform(X_resampled)

    # ===============================
    # 5. TRAIN MODEL
    # ===============================
    st.markdown("<div class='section-title'>📈 Model Training</div>", unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_resampled, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    fig3 = plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig3)

    # ===============================
    # 6. PREDICTION FORM
    # ===============================
    st.markdown("<div class='section-title'>🔮 Predict Fraud for New Transaction</div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        amt = st.number_input("Transaction Amount", 0.0, 20000.0, 150.0)
        age = st.number_input("Account Age (Days)", 0, 5000, 300)
        risk = st.number_input("Risk Score", 0.0, 100.0, 10.0)
        hr = st.slider("Hour of Transaction", 0, 23, 12)

    with colB:
        day = st.selectbox("Day of Week", sorted(df["DayOfWeek"].unique()))
        cat = st.selectbox("Merchant Category", sorted(df["MerchantCategory"].unique()))
        mode = st.selectbox("Transaction Mode", sorted(df["TransactionMode"].unique()))
        loc = st.selectbox("Location", sorted(df["Location"].unique()))

    if st.button("Predict Fraud"):
        new_data = pd.DataFrame([{
            "TransactionAmount": amt,
            "AccountAgeDays": age,
            "RiskScore": risk,
            "Hour": hr,
            "MerchantCategory": cat,
            "TransactionMode": mode,
            "Location": loc,
            "DayOfWeek": day
        }])

        encoded_new = pre_encoder.transform(new_data)
        prediction = model.predict(encoded_new)[0]

        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Normal Transaction")

else:
    st.info("📂 Upload your dataset to start.")
