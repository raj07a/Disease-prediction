# disease_prediction.py (Upgraded Version for "Uses of AI in Healthcare for Disease Prediction")

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
# -------------------- Step 1: Load Datasets --------------------
symptom_df = pd.read_csv("DiseaseAndSymptoms.csv")
precaution_df = pd.read_csv("Disease precaution.csv")




# -------------------- Step 2: Preprocessing --------------------
symptom_cols = [col for col in symptom_df.columns if col.startswith("Symptom")]
symptom_df[symptom_cols] = symptom_df[symptom_cols].fillna("")
symptom_df['all_symptoms'] = symptom_df[symptom_cols].values.tolist()
symptom_df['all_symptoms'] = symptom_df['all_symptoms'].apply(lambda x: sorted(set(sym.strip().lower() for sym in x if sym.strip() != "")))

# Handle duplicates
symptom_df['all_symptoms_str'] = symptom_df['all_symptoms'].apply(lambda x: ','.join(x))
symptom_df.drop_duplicates(subset=['Disease', 'all_symptoms_str'], inplace=True)

# -------------------- Step 3: Balance the Dataset --------------------
from sklearn.utils import resample
min_count = symptom_df['Disease'].value_counts().min()
balanced_df = pd.concat([
    group.sample(min_count, random_state=42)
    for _, group in symptom_df.groupby('Disease')
])

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(balanced_df['all_symptoms'])

le = LabelEncoder()
y = le.fit_transform(balanced_df['Disease'])

# -------------------- Step 4: Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Step 5: Model Training --------------------
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# -------------------- Step 6: Evaluation --------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(le.classes_)),
    target_names=le.classes_,
    zero_division=0
))

# -------------------- Step 7: Save the Model --------------------
joblib.dump(model, "disease_model.pkl")
joblib.dump(mlb, "symptom_encoder.pkl")
joblib.dump(le, "disease_encoder.pkl")

# -------------------- Step 8: Utility - Prediction Function --------------------
def predict_disease(symptom_list):
    model = joblib.load("disease_model.pkl")
    mlb = joblib.load("symptom_encoder.pkl")
    le = joblib.load("disease_encoder.pkl")

    processed_input = sorted(set(sym.strip().lower() for sym in symptom_list if sym.strip() != ""))
    input_vector = mlb.transform([processed_input])
    pred_proba = model.predict_proba(input_vector)[0]
    pred_index = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_index])[0]
    confidence = round(pred_proba[pred_index] * 100, 2)

    return disease, confidence

# -------------------- Step 9: Utility - Get Precautions --------------------
def get_precautions(disease_name):
    row = precaution_df[precaution_df['Disease'].str.lower() == disease_name.lower()]
    if row.empty:
        return ["No precautions found."]
    return row.iloc[0, 1:].dropna().tolist()
# -------------------- Step 10: Streamlit UI (Enhanced + Larger Text UI) --------------------
st.set_page_config(page_title="Disease Predictor AI", layout="wide")

# Custom CSS for font sizes and padding
st.markdown("""
    <style>
    h1, h2, h3, .big-text {
        font-size: 28px !important;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .stMultiSelect>div>div>div>div {
        font-size: 18px;
    }
    .stButton>button {
        font-size: 18px;
        padding: 0.5em 2em;
    }
    .stDataFrame {
        font-size: 16px;
    }
    .stAlert>div {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color:#2E86C1;'>🤖 AI-Powered Disease Predictor</h1>", unsafe_allow_html=True)

st.markdown("### 🩺 Select your symptoms from the list below:")
valid_symptoms = list(mlb.classes_)
user_symptoms = st.multiselect("🧬 Symptoms", options=valid_symptoms)

# Basic dictionary (can expand)
disease_descriptions = {
    "diabetes": "A chronic condition that affects the way your body processes blood sugar (glucose).",
    "migraine": "A neurological condition that can cause multiple symptoms, often intense headaches.",
    "hypertension": "Also known as high blood pressure, it can lead to heart problems if untreated.",
    "arthritis": "Inflammation of joints causing pain and stiffness, typically worsens with age.",
    "common cold": "A viral infectious disease of the upper respiratory tract."
}

# Risk calculation
def get_risk_level(num_symptoms):
    if num_symptoms >= 6:
        return "High"
    elif num_symptoms >= 3:
        return "Moderate"
    else:
        return "Low"

# -------------------- Step 11: Prediction --------------------
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🔍 Predict Disease"):
        if user_symptoms:
            disease, confidence = predict_disease(user_symptoms)
            precautions = get_precautions(disease)
            description = disease_descriptions.get(disease.lower(), "No description available.")
            risk = get_risk_level(len(user_symptoms))

            # Save to history
            st.session_state.history.append({
                "Symptoms": ", ".join(user_symptoms),
                "Disease": disease,
                "Confidence": f"{confidence}%",
                "Risk Level": risk
            })

            st.markdown(f"<h3 style='color:green;'>🧠 Predicted Disease: <b>{disease}</b></h3>", unsafe_allow_html=True)
            st.progress(confidence / 100.0)
            st.markdown(f"<div class='big-text'>📈 <b>Confidence:</b> {confidence}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-text'>🔶 <b>Risk Level:</b> {risk}</div>", unsafe_allow_html=True)

            with st.expander("📘 Disease Description"):
                st.write(description)

            with st.expander("⚠️ Recommended Precautions"):
                for i, p in enumerate(precautions, 1):
                    st.markdown(f"{i}. {p}")
        else:
            st.warning("⚠️ Please select at least one symptom to get prediction.")

with col2:
    if st.button("🔄 Reset"):
        st.session_state.history = []
        st.experimental_rerun()

# -------------------- Step 12: Frequency Plot --------------------
st.markdown("### 📊 Top 15 Most Frequent Diseases in Dataset")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=symptom_df, y="Disease", order=symptom_df["Disease"].value_counts().index[:15], palette="coolwarm", ax=ax)
ax.set_title("Disease Frequency in Dataset", fontsize=16)
st.pyplot(fig)

# -------------------- Step 13: Prediction History Table --------------------
if st.session_state.history:
    st.markdown("### 🧾 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

# -------------------- Step 14: Footer --------------------
st.markdown("""
---
<div style='font-size:16px;'>
Developed with ❤️ using <b>Streamlit</b>  
<br>Author: <b>RAJA RAWAT</b> | Research Intern @ <b>SPSU</b>
</div>
""", unsafe_allow_html=True)
