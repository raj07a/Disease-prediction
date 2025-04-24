import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os

# -------------------- Load Model and Encoders --------------------
model = joblib.load("disease_model.pkl")
mlb = joblib.load("symptom_encoder.pkl")
le = joblib.load("disease_encoder.pkl")

# -------------------- Load Metadata --------------------
precaution_df = pd.read_csv("Disease precaution.csv")
desc_df = pd.read_csv("description.csv")
desc_df["Disease"] = desc_df["Disease"].str.strip().str.lower()
desc_dict = dict(zip(desc_df["Disease"], desc_df["Description"]))

# -------------------- Utility Functions --------------------
def predict_disease(symptom_list):
    processed = sorted(set(s.strip().lower() for s in symptom_list if s.strip()))
    input_vector = mlb.transform([processed])
    proba = model.predict_proba(input_vector)[0]
    pred_idx = np.argmax(proba)
    disease = le.inverse_transform([pred_idx])[0]
    confidence = round(proba[pred_idx] * 100, 2)
    return disease, confidence

def get_precautions(disease_name):
    row = precaution_df[precaution_df['Disease'].str.lower() == disease_name.lower()]
    if row.empty:
        return ["No specific precautions listed."]
    return row.iloc[0, 1:].dropna().tolist()

def get_risk_level(num_symptoms):
    if num_symptoms >= 6:
        return "High"
    elif num_symptoms >= 3:
        return "Moderate"
    else:
        return "Low"

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Disease Predictor Chat", layout="centered")
st.markdown("""
    <style>
        h1, h2, h3, .stTextInput, .stSelectbox label, .stMarkdown {
            font-size: 18px !important;
        }
        .stChatMessage, .stChatInput input {
            font-size: 18px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color:#2E86C1;'>💬 Chat with AI Disease Predictor</h2>", unsafe_allow_html=True)
st.markdown("Enter symptoms like: 'I have fever, cough, chills'")

# -------------------- Age & Gender Inputs --------------------
col1, col2 = st.columns(2)
with col1:
    age_group = st.selectbox("👤 Age Group", ["< 18", "18 - 35", "36 - 60", "60+"], key="age")
with col2:
    gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"], key="gender")

# -------------------- Initialize Chat Session --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# -------------------- Chat Input Box --------------------
user_input = st.chat_input("Describe your symptoms...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Extract valid symptoms
    raw_symptoms = re.split(",| and |\.|\n", user_input.lower())
    matched_symptoms = [s.strip() for s in raw_symptoms if s.strip() in mlb.classes_]

    if not matched_symptoms:
        response = "❌ I couldn't recognize any valid symptoms. Please rephrase using common medical terms."
    else:
        disease, confidence = predict_disease(matched_symptoms)
        risk = get_risk_level(len(matched_symptoms))
        description = desc_dict.get(disease.lower(), "No description available.")
        precautions = get_precautions(disease)

        response = f"""
🧠 **Predicted Disease**: `{disease}`  
📊 **Confidence**: `{confidence}%`  
🔶 **Risk Level**: `{risk}`  
👤 **Age Group**: `{age_group}`  
⚧ **Gender**: `{gender}`  

📘 **Description**:
> {description}

⚠️ **Precautions**:
""" + "\n".join([f"- {p}" for p in precautions])

    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
