import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Waveform Model Comparison", layout="wide")
st.title("Waveform Generator")

# --- Load dataset ---
@st.cache_data
def load_data():
    return pd.read_csv('waveform.csv')

df = load_data()

# --- Prepare data ---
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Define models ---
model_options = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42),
    "Gradient Boosting (XGBoost)": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

selected_models = st.multiselect(
    "Select models to evaluate:", list(model_options.keys()), default=[]
)

run_button = st.button("🚀 Run Evaluation")

if run_button and selected_models:
    results = []

    for name in selected_models:
        model = model_options[name]
        use_scaled = name in ["Support Vector Machine", "Logistic Regression", "K-Nearest Neighbors"]

        X_train_input = X_train_scaled if use_scaled else X_train
        X_test_input = X_test_scaled if use_scaled else X_test

        model.fit(X_train_input, y_train)
        y_pred = model.predict(X_test_input)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        results.append((name, accuracy, precision))

        st.subheader(f" Classification Report: {name}")
        target_names = ["Waveform 1", "Waveform 2", "Waveform 3"]
        print(classification_report(y_test, y_pred, target_names=target_names))
        st.text(classification_report(y_test, y_pred))

        st.subheader(f" Confusion Matrix: {name}")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        st.pyplot(fig)

    # --- Summary ---
    result_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision'])
    st.subheader("📈 Model Comparison Summary")
    st.dataframe(result_df.sort_values(by="Accuracy", ascending=False), use_container_width=True)

    st.subheader("📊 Accuracy & Precision Plots")
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    sns.barplot(data=result_df, x='Accuracy', y='Model', palette='viridis', ax=axs[0])
    axs[0].set_xlim(0, 1)
    axs[0].set_title("Accuracy")

    sns.barplot(data=result_df, x='Precision', y='Model', palette='magma', ax=axs[1])
    axs[1].set_xlim(0, 1)
    axs[1].set_title("Precision")

    st.pyplot(fig)
elif run_button:
    st.warning("Please select at least one model to run the evaluation.")
