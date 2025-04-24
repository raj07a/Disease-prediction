import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and encoders
model = joblib.load("disease_model.pkl")
mlb = joblib.load("symptom_encoder.pkl")
le = joblib.load("disease_encoder.pkl")
precaution_df = pd.read_csv("Disease precaution.csv")
symptom_df = pd.read_csv("DiseaseAndSymptoms.csv")

disease_descriptions = {
    "aids": "AIDS weakens the immune system by destroying important cells that fight disease and infection.",
    "acne": "A skin condition that occurs when hair follicles become clogged with oil and dead skin cells.",
    "migraine": "A neurological condition causing intense, debilitating headaches.",
    "diabetes": "A metabolic disease that causes high blood sugar levels over a prolonged period.",
    "arthritis": "Inflammation of joints causing pain and stiffness.",
    # ... Add all 41 descriptions
}

# Helper functions
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
    return row.iloc[0, 1:].dropna().tolist() if not row.empty else ["No precautions found."]

def get_risk_level(num_symptoms):
    if num_symptoms >= 6:
        return "High"
    elif num_symptoms >= 3:
        return "Moderate"
    else:
        return "Low"

# Streamlit UI setup
st.set_page_config(page_title="AI Disease Chatbot", layout="centered")
st.title("💬 AI-Powered Disease Prediction Chatbot")
st.markdown("Describe your symptoms below. Example: 'I have headache, fever and chills'")

col1, col2 = st.columns(2)
with col1:
    age_group = st.selectbox("👤 Age Group", ["< 18", "18 - 35", "36 - 60", "60+"])
with col2:
    gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])

# Chat session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Enter your symptoms here...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Extract symptoms from input
    raw_symptoms = re.split(",| and |\n", user_input.lower())
    matched_symptoms = [s.strip() for s in raw_symptoms if s.strip() in mlb.classes_]

    if not matched_symptoms:
        response = "❌ Sorry, I couldn't recognize any valid symptoms. Try using standard medical terms."
    else:
        disease, confidence = predict_disease(matched_symptoms)
        description = disease_descriptions.get(disease.lower(), "No description available.")
        precautions = get_precautions(disease)
        risk = get_risk_level(len(matched_symptoms))

        response = f"""
### 🧠 Predicted Disease: **{disease}**
- 📈 Confidence: **{confidence}%**
- 🔶 Risk Level: **{risk}**
- 👤 Age Group: **{age_group}**
- ⚧ Gender: **{gender}**

**📘 Description:**
{description}

**⚠️ Precautions:**
""" + "\n".join([f"- {p}" for p in precautions])

    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Disease Frequency Plot
st.markdown("---")
st.subheader("📊 Top 15 Diseases in Dataset")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=symptom_df, y="Disease", order=symptom_df["Disease"].value_counts().index[:15], palette="coolwarm", ax=ax)
ax.set_title("Most Common Diseases", fontsize=16)
st.pyplot(fig)

# Footer
st.markdown("""
---
Developed with ❤️ using Streamlit  
Author: **Your Name** | Research Intern @ **SPSU**
""")
