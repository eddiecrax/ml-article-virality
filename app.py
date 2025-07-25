import streamlit as st
import pandas as pd
import joblib

# Load your trained model and features list
model = joblib.load("random_forest_model.pkl")  # replace with your model path
feature_names = joblib.load("trained_features.pkl")  # features used during training

viral_keywords = [
    'died', 'death', 'war', 'ukraine', 'israel', 'russia', 'elections',
    'scandal', 'breaking', 'president', 'celebrity', 'fame', 'explosion', 'attack', 'murder'
]

def contains_viral_terms(text):
    text = str(text).lower()
    return int(any(kw in text for kw in viral_keywords))

st.title("Viral News Article Predictor")

headline = st.text_input("Enter a news headline:")

if headline:
    # Create DataFrame with one sample
    input_df = pd.DataFrame([{
        'has_viral_terms': contains_viral_terms(headline),
        # For demo: add other numeric features with default 0 or mean values
        # Adjust as per your model features!
        **{feat: 0 for feat in feature_names if feat != 'has_viral_terms'}
    }])

    # Align columns exactly to what model expects
    input_df = input_df[feature_names]

    # Predict
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.write(f"**Prediction:** {'Viral' if pred == 1 else 'Not Viral'}")
    st.write(f"**Probability of going viral:** {proba:.2%}")

# Run this app by saving and running:
# streamlit run streamlit_app.py
