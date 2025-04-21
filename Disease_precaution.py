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
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# Load model and encoders
disease_model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")
disease_metadata = joblib.load("disease_metadata.pkl")

# Load all symptoms from the encoder
all_symptoms = list(symptom_encoder.classes_)

# Streamlit UI
st.set_page_config(page_title="AI Disease Predictor", layout="centered")
st.title("🤖 AI-Powered Disease Predictor")
st.markdown("Select your symptoms below from the list.")

# Symptom selection
selected_symptoms = st.multiselect("Select Symptoms:", all_symptoms)

# Predict button
if st.button("🔍 Predict Disease"):
    if selected_symptoms:
        # Prepare input for model
        input_vector = symptom_encoder.transform([selected_symptoms])
        prediction = disease_model.predict(input_vector)[0]
        confidence = max(disease_model.predict_proba(input_vector)[0]) * 100
        predicted_disease = label_encoder.inverse_transform([prediction])[0].lower()

        # Show Prediction
        st.success(f"🧾 Predicted Disease: {predicted_disease.title()}")
        st.info(f"📈 Confidence: {confidence:.2f}%")

        # Fetch metadata
        info = disease_metadata.get(predicted_disease, {})

        if info:
            st.markdown(f"### 📘 Description")
            st.write(info['description'])

            st.markdown("### ⚠️ Precautions")
            for i, p in enumerate(info['precautions'], 1):
                st.markdown(f"{i}. {p}")

            st.markdown("### 💊 Medications")
            for m in info['medications']:
                st.markdown(f"- {m}")

            st.markdown("### 🥗 Diet Plan")
            for d in info['diet']:
                st.markdown(f"- {d}")

            st.markdown("### 🏃 Recommended Workout")
            for w in info['workout']:
                st.markdown(f"- {w}")
        else:
            st.warning("ℹ️ No extra information found for this disease.")
    else:
        st.warning("Please select at least one symptom to proceed.")
