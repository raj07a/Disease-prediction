import pandas as pd
import numpy as np
import re
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fuzzywuzzy import process
# -------------------- Step 1: Load and Preprocess Data --------------------
symptom_df = pd.read_csv("DiseaseAndSymptoms.csv")
precaution_df = pd.read_csv("Disease precaution.csv")

symptom_cols = [col for col in symptom_df.columns if col.startswith("Symptom")]
symptom_df[symptom_cols] = symptom_df[symptom_cols].fillna("")
symptom_df["all_symptoms"] = symptom_df[symptom_cols].values.tolist()
symptom_df["all_symptoms"] = symptom_df["all_symptoms"].apply(
    lambda x: sorted(set(sym.strip().lower() for sym in x if sym.strip() != ""))
)
symptom_df["all_symptoms_str"] = symptom_df["all_symptoms"].apply(lambda x: ",".join(x))
symptom_df.drop_duplicates(subset=["Disease", "all_symptoms_str"], inplace=True)

# Balance
df_grouped = symptom_df.groupby("Disease")
min_count = df_grouped.size().min()
balanced_df = pd.concat(
    [grp.sample(min_count, random_state=42) for _, grp in df_grouped]
)

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(balanced_df["all_symptoms"])
le = LabelEncoder()
y = le.fit_transform(balanced_df["Disease"])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# -------------------- Step 2: Utilities --------------------
def predict_disease(symptoms):
    processed = sorted(set(sym.strip().lower() for sym in symptoms if sym.strip() != ""))
    input_vector = mlb.transform([processed])
    probs = model.predict_proba(input_vector)[0]
    idx = np.argmax(probs)
    disease = le.inverse_transform([idx])[0]
    confidence = round(probs[idx] * 100, 2)
    return disease, confidence

def get_precautions(disease_name):
    row = precaution_df[precaution_df["Disease"].str.lower() == disease_name.lower()]
    if row.empty:
        return ["No precautions found."]
    return row.iloc[0, 1:].dropna().tolist()

def get_risk_level(symptom_count):
    if symptom_count >= 6:
        return "High"
    elif symptom_count >= 3:
        return "Moderate"
    else:
        return "Low"

# Sample Descriptions
disease_descriptions = {
    "common cold": "A viral upper respiratory tract infection causing sneezing and sore throat.",
    "diabetes": "A condition affecting blood sugar regulation over time.",
    "migraine": "A neurological disorder causing recurrent intense headaches.",
    "arthritis": "Joint inflammation leading to pain and stiffness.",
    "allergy": "Immune response to a foreign substance such as pollen or dust."
    # Add more as needed...
}

# -------------------- Step 3: Streamlit Chatbot UI --------------------
st.set_page_config("AI Disease Chatbot", layout="centered")

st.markdown("""
    <style>
    .chat-entry {font-size: 18px;}
    .stTextInput input {font-size: 18px;}
    .stSelectbox > div > div > div {font-size: 18px;}
    .stButton button {font-size: 18px; padding: 0.4em 1em;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>💬 AI Disease Prediction Chatbot</h1>", unsafe_allow_html=True)
st.markdown("Describe your symptoms below, e.g., 'I have headache and chills'")

col1, col2 = st.columns(2)
with col1:
    age_group = st.selectbox("👤 Age Group", ["< 18", "18 - 35", "36 - 60", "60+"])
with col2:
    gender = st.selectbox("♇ Gender", ["Male", "Female", "Other"])

# Session for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Type your symptoms here...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    raw_symptoms = re.split(",| and |\n", user_input.lower())
valid_symptoms = list(mlb.classes_)

matched_symptoms = []
for s in raw_symptoms:
    match = fuzzy_match(s.strip(), valid_symptoms)
    if match:
        matched_symptoms.append(match)

if not matched_symptoms:
    response = "❌ I couldn't detect any valid symptoms. Please use more recognizable medical terms."
else:
    disease, confidence = predict_disease(matched_symptoms)
    description = disease_descriptions.get(disease.lower(), "No description available.")
    precautions = get_precautions(disease)
    risk = get_risk_level(len(matched_symptoms))

    response = f"""

### 🧠 Predicted Disease: {disease}
- 📈 Confidence: {confidence}%
- 🔶 Risk Level: {risk}
- 👤 Age Group: {age_group}
- ♇ Gender: {gender}

📘 Description:
{description}

⚠️ Precautions:
""" + "\n".join([f"- {p}" for p in precautions])

    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# -------------------- Step 4: Chart --------------------
st.markdown("---")
st.subheader("📊 Top 15 Diseases in Dataset")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=symptom_df, y="Disease", order=symptom_df["Disease"].value_counts().index[:15], palette="coolwarm", ax=ax)
ax.set_title("Most Common Diseases", fontsize=16)
st.pyplot(fig)

st.markdown("""
---
<small>Developed by ❤️ with Streamlit  
<b>Research Intern @ SPSU</b></small>
""", unsafe_allow_html=True)

