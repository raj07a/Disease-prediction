import pandas as pd

# Load datasets
old_symptom_df = pd.read_csv("DiseaseAndSymptoms.csv")
old_precaution_df = pd.read_csv("Disease precaution.csv")
diet_df = pd.read_csv("diets.csv")
med_df = pd.read_csv("medications.csv")
workout_df = pd.read_csv("workout_df.csv")
description_df = pd.read_csv("description.csv")
new_precaution_df = pd.read_csv("precautions_df.csv")

# Clean function
def clean_disease_names(df, column):
    df[column] = df[column].str.strip().str.lower()
    return df

# Apply cleaning
old_symptom_df = clean_disease_names(old_symptom_df, "Disease")
old_precaution_df = clean_disease_names(old_precaution_df, "Disease")
diet_df = clean_disease_names(diet_df, "Disease")
med_df = clean_disease_names(med_df, "Disease")
workout_df = clean_disease_names(workout_df, "disease")
description_df = clean_disease_names(description_df, "Disease")
new_precaution_df = clean_disease_names(new_precaution_df, "Disease")

# Build metadata dictionary
disease_metadata = {}

for disease in old_symptom_df["Disease"].unique():
    dis = disease.lower().strip()
    disease_metadata[dis] = {
        "precautions": (
            old_precaution_df[old_precaution_df["Disease"] == dis].iloc[:, 1:].dropna(axis=1).values.tolist()[0]
            if dis in old_precaution_df["Disease"].values else
            new_precaution_df[new_precaution_df["Disease"] == dis].iloc[:, 1:].dropna(axis=1).values.tolist()[0]
            if dis in new_precaution_df["Disease"].values else []
        ),
        "medications": (
            med_df[med_df["Disease"] == dis].iloc[:, 1:].dropna(axis=1).values.tolist()[0]
            if dis in med_df["Disease"].values else []
        ),
        "diet": (
            diet_df[diet_df["Disease"] == dis].iloc[:, 1:].dropna(axis=1).values.tolist()[0]
            if dis in diet_df["Disease"].values else []
        ),
        "workout": (
            workout_df[workout_df["disease"] == dis].iloc[:, 1:].dropna(axis=1).values.tolist()[0]
            if dis in workout_df["disease"].values else []
        ),
        "description": (
            description_df[description_df["Disease"] == dis]["Description"].values[0]
            if dis in description_df["Disease"].values else ""
        )
    }

# Save the metadata for use in your app
import joblib

joblib.dump(disease_metadata, "disease_metadata.pkl")

import streamlit as st
import joblib

# Load the saved metadata
disease_metadata = joblib.load("disease_metadata.pkl")

# Example predicted disease (this should come from your ML model in actual use)
predicted_disease = "diabetes"
info = disease_metadata.get(predicted_disease.lower(), {})

# Set page config and style
st.set_page_config(page_title="Disease Information", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
        padding: 20px;
    }
    .section-title {
        font-size: 22px;
        color: #2c3e50;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .content-block {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Disease Insights")

st.subheader(f"🔍 Disease Predicted: `{predicted_disease.title()}`")

# Description Section
st.markdown("### 📘 Description")
st.markdown(f"<div class='content-block'>{info.get('description', 'No description available.')}</div>", unsafe_allow_html=True)

# Columns for multiple sections
col1, col2 = st.columns(2)

# Precautions
with col1:
    st.markdown("### ⚠️ Precautions")
    if info.get("precautions"):
        for i, p in enumerate(info["precautions"], 1):
            st.markdown(f"- {p}")
    else:
        st.markdown("No precautions available.")

# Medications
with col2:
    st.markdown("### 💊 Medications")
    if info.get("medications"):
        for m in info["medications"]:
            st.markdown(f"- {m}")
    else:
        st.markdown("No medications listed.")

# Diet Plan
st.markdown("### 🥗 Diet Plan")
if info.get("diet"):
    st.markdown("<div class='content-block'>" + "".join(f"<li>{d}</li>" for d in info["diet"]) + "</div>", unsafe_allow_html=True)
else:
    st.markdown("No diet plan available.")

# Workout
st.markdown("### 🏃 Recommended Workout")
if info.get("workout"):
    st.markdown("<div class='content-block'>" + "".join(f"<li>{w}</li>" for w in info["workout"]) + "</div>", unsafe_allow_html=True)
else:
    st.markdown("No workout recommendation available.")

