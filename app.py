import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    return joblib.load('isolation_forest_model1.pkl')

def detect_fraud(transactions, model):
    features = transactions.drop(columns=['Class'])  # Drop 'Class' if it's included
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    predictions = model.predict(features_scaled)
    fraud_transactions = transactions[predictions == -1]
    return fraud_transactions

def main():
    st.title("Credit Card Fraud Detection")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            transactions = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            transactions = pd.read_excel(uploaded_file, sheet_name='creditcard_test')

        st.write("Dataset Preview:")
        st.write(transactions.head())

        if st.button("Run Anomaly Detection"):
            model = load_model()
            fraud_transactions = detect_fraud(transactions, model)
            st.write(f"Number of Fraudulent Transactions Detected: {len(fraud_transactions)}")
            st.write(fraud_transactions)
            
            st.write("Anomaly Visualization:")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=transactions.columns[0], y=transactions.columns[1], hue=(fraud_transactions.index), palette='coolwarm', alpha=0.6)
            plt.title("Detected Anomalies in Transactions")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            st.pyplot(plt)

if __name__ == "__main__":
    main()
