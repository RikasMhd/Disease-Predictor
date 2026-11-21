import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = "disease_model.pkl"
DATA_PATH = "medical_dataset.csv"

st.set_page_config(page_title="Disease Predictor", page_icon="🩺")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["disease"] = df["disease"].str.replace("\u2019", "'", regex=False).str.strip()
    return df


@st.cache_resource
def load_model():
    # Saved model irundha adhai use pannum, illana CSV-la irundhu train pannum
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    df = load_data()
    X = (
        df[["symptom_1", "symptom_2", "symptom_3"]]
        .fillna("").astype(str).agg(" ".join, axis=1)
        .str.replace("_", " ").str.lower()
    )
    model = Pipeline([
        ("vectorizer", CountVectorizer(binary=True)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X, df["disease"])
    return model


@st.cache_data
def load_symptoms():
    df = load_data()
    s = (
        df[["symptom_1", "symptom_2", "symptom_3"]]
        .stack().astype(str).str.replace("_", " ").str.lower().str.strip()
    )
    return sorted(s.unique())


model = load_model()
symptoms = load_symptoms()

st.title("🩺 Symptom-Based Disease Predictor")
st.warning(
    "⚠️ Educational machine learning project on a small dataset (82 records). "
    "This is NOT a medical diagnosis tool. Please consult a doctor for health concerns."
)

st.write("Select exactly **3 symptoms** and click Predict.")
chosen = st.multiselect("Symptoms", symptoms, max_selections=3)

if st.button("Predict", type="primary"):
    if len(chosen) != 3:
        st.error("Please select exactly 3 symptoms.")
    else:
        text = " ".join(chosen)
        probs = model.predict_proba([text])[0]
        top3 = np.argsort(probs)[::-1][:3]

        st.success(f"Most likely: **{model.classes_[top3[0]]}** ({probs[top3[0]] * 100:.1f}%)")
        st.subheader("Top 3 possibilities")
        for i in top3:
            st.write(f"{model.classes_[i]} — {probs[i] * 100:.1f}%")
            st.progress(float(probs[i]))

st.caption("Model: CountVectorizer + Logistic Regression (scikit-learn)")