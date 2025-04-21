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

import joblib
import streamlit as st
disease_metadata = joblib.load("disease_metadata.pkl")

predicted_disease = "diabetes"  # This will come from your ML model
info = disease_metadata.get(predicted_disease.lower(), {})

st.markdown(f"### 🧠 Disease Info: {predicted_disease.title()}")
st.write(f"📘 Description: {info['description']}")
st.write("⚠️ Precautions:")
for i, p in enumerate(info['precautions'], 1):
    st.markdown(f"{i}. {p}")
st.write("💊 Medications:")
for m in info['medications']:
    st.markdown(f"- {m}")
st.write("🥗 Diet Plan:")
for d in info['diet']:
    st.markdown(f"- {d}")
st.write("🏃 Recommended Workout:")
for w in info['workout']:
    st.markdown(f"- {w}")
