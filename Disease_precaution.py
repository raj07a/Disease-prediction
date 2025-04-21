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

# Save the metadata for use in your app
import streamlit as st
import joblib

# Load model & encoders
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")
disease_metadata = joblib.load("disease_metadata.pkl")

# Page config
st.set_page_config("Disease Predictor AI", layout="centered")
st.title("🧠 Disease Prediction & Medical Advisor")

# Multiselect
st.subheader("🩺 Select Symptoms")
all_symptoms = list(symptom_encoder.classes_)
selected_symptoms = st.multiselect("Choose symptoms you're experiencing:", options=all_symptoms)

if st.button("🔍 Predict"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Predict
        input_vector = symptom_encoder.transform([selected_symptoms])
        proba = model.predict_proba(input_vector)[0]
        pred_index = proba.argmax()
        predicted_disease = label_encoder.inverse_transform([pred_index])[0]
        confidence = round(proba[pred_index] * 100, 2)

        # Get disease info
        info = disease_metadata.get(predicted_disease.lower(), {})

        st.success(f"🧠 Predicted Disease: **{predicted_disease}**")
        st.progress(confidence / 100.0, text=f"Confidence: {confidence}%")

        if info:
            # Description
            if info.get("description"):
                with st.expander("📘 Description"):
                    st.write(info["description"])

            # Precautions
            if info.get("precautions"):
                with st.expander("⚠️ Precautions"):
                    for i, p in enumerate(info["precautions"], 1):
                        st.markdown(f"{i}. {p}")

            # Medications
            if info.get("medications"):
                with st.expander("💊 Medications"):
                    meds = [m for m in info["medications"] if isinstance(m, str)]
                    for m in meds:
                        st.markdown(f"- {m}")

            # Diet
            if info.get("diet"):
                with st.expander("🥗 Diet Recommendations"):
                    for d in info["diet"]:
                        st.markdown(f"- {d}")

            # Workout
            if info.get("workout"):
                with st.expander("🏃 Workout Suggestions"):
                    workout_cleaned = list({w for w in info["workout"] if isinstance(w, str) and len(w) > 3 and predicted_disease.lower() not in w.lower()})
                    for w in workout_cleaned:
                        st.markdown(f"- {w}")
        else:
            st.warning("No metadata found for this disease.")

