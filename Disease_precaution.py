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
import numpy as np

# Load models and metadata
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")
disease_metadata = joblib.load("disease_metadata.pkl")

# App Configuration
st.set_page_config("AI-Powered Disease Predictor", layout="centered", page_icon="🧠")
st.markdown("""
    <h1 style='text-align: center; color: white;'>🤖 AI-Powered Disease Predictor</h1>
    <p style='text-align: center; color: gray;'>Select your symptoms below from the list or enter manually.</p>
""", unsafe_allow_html=True)

# --- UI Styling ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: 1px solid red;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #ff0000;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- User Input ---
all_symptoms = list(symptom_encoder.classes_)
selected_symptoms = st.multiselect("Select Symptoms:", options=all_symptoms)

manual_input = st.text_input("Or type symptoms manually (comma separated):", placeholder="e.g. headache, fatigue")

# Merge selected and typed symptoms
if manual_input:
    typed = [s.strip().lower().replace(" ", "_") for s in manual_input.split(",")]
    selected_symptoms += [s for s in typed if s in all_symptoms]

selected_symptoms = list(set(selected_symptoms))  # remove duplicates

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select or enter at least one valid symptom.")
    else:
        try:
            input_vector = symptom_encoder.transform([selected_symptoms])
            proba = model.predict_proba(input_vector)[0]
            pred_index = np.argmax(proba)
            predicted_disease = label_encoder.inverse_transform([pred_index])[0]
            confidence = round(proba[pred_index] * 100, 2)

            info = disease_metadata.get(predicted_disease.lower(), {})

            st.success(f"🧠 **Predicted Disease:** {predicted_disease.title()}")
            st.info(f"**Confidence:** {confidence}%")

            # Description
            if info.get("description"):
                st.subheader("📘 Description")
                st.write(info["description"])

            # Precautions
            if info.get("precautions"):
                st.subheader("⚠️ Precautions to Take")
                for i, p in enumerate(info["precautions"], 1):
                    st.markdown(f"{i}. {p}")

            # Medications
            if info.get("medications"):
                st.subheader("💊 Recommended Medications")
                for m in info["medications"]:
                    st.markdown(f"- {m}")

            # Diet
            if info.get("diet"):
                st.subheader("🥗 Suggested Diet")
                for d in info["diet"]:
                    st.markdown(f"- {d}")

            # Workout
            if info.get("workout"):
                st.subheader("🏃 Workout Suggestions")
                workout_cleaned = [w for w in info["workout"] if isinstance(w, str) and len(w) > 3]
                for w in workout_cleaned:
                    st.markdown(f"- {w}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")


