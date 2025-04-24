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


# -------------------- Step 10: Streamlit UI (Final Professional Version) --------------------
st.set_page_config(page_title="Disease Predictor AI", layout="centered")

# Manual description dictionary (based on 41 diseases)
disease_descriptions = {
    "aids": "AIDS weakens the immune system by destroying important cells that fight disease and infection.",
    "acne": "A skin condition that occurs when hair follicles become clogged with oil and dead skin cells.",
    "alcoholic hepatitis": "Liver inflammation caused by excessive alcohol consumption.",
    "allergy": "A condition in which the immune system reacts abnormally to a foreign substance.",
    "arthritis": "Inflammation of joints causing pain and stiffness.",
    "bronchial asthma": "A respiratory condition marked by spasms in the bronchi of the lungs.",
    "cervical spondylosis": "Age-related wear and tear affecting the spinal disks in your neck.",
    "chicken pox": "A highly contagious viral infection causing an itchy, blister-like rash.",
    "chronic cholestasis": "Reduced bile flow from the liver leading to accumulation of bile acids.",
    "common cold": "A viral upper respiratory tract infection causing sneezing, sore throat, and congestion.",
    "dengue": "A mosquito-borne viral infection causing high fever, rash, and muscle pain.",
    "diabetes": "A metabolic disease that causes high blood sugar levels over a prolonged period.",
    "dimorphic hemmorhoids(piles)": "Swollen veins in the anus and lower rectum causing discomfort and bleeding.",
    "drug reaction": "An abnormal response of the body to a medication.",
    "fungal infection": "Diseases caused by fungi that can affect skin, nails, lungs, or internal organs.",
    "gerd": "Gastroesophageal reflux disease - stomach acid frequently flows back into the esophagus.",
    "gastroenteritis": "Inflammation of the stomach and intestines causing diarrhea, vomiting, and cramps.",
    "heart attack": "A blockage of blood flow to the heart muscle.",
    "hepatitis b": "A serious liver infection caused by the hepatitis B virus.",
    "hepatitis c": "A viral infection that causes liver inflammation, sometimes leading to serious damage.",
    "hepatitis d": "A liver infection caused by the hepatitis D virus; occurs only in those infected with HBV.",
    "hepatitis e": "A liver disease caused by the hepatitis E virus, often spread via contaminated water.",
    "hypertension": "A condition in which the force of the blood against the artery walls is too high.",
    "hyperthyroidism": "Overproduction of hormones by the thyroid gland.",
    "hypoglycemia": "A condition caused by a very low level of blood sugar (glucose).",
    "hypothyroidism": "A condition in which the thyroid gland doesn't produce enough hormones.",
    "impetigo": "A highly contagious skin infection causing red sores, mostly in children.",
    "jaundice": "A liver condition that causes yellowing of skin and eyes due to excess bilirubin.",
    "malaria": "A disease caused by a plasmodium parasite, transmitted by mosquito bites.",
    "migraine": "A neurological condition causing intense, debilitating headaches.",
    "osteoarthristis": "Degeneration of joint cartilage and the underlying bone, common in older people.",
    "paralysis (brain hemorrhage)": "Loss of muscle function due to brain damage from bleeding.",
    "peptic ulcer diseae": "Sores that develop on the inside lining of your stomach or small intestine.",
    "pneumonia": "Infection that inflames air sacs in one or both lungs, possibly with fluid.",
    "psoriasis": "A chronic autoimmune condition that causes the rapid buildup of skin cells.",
    "tuberculosis": "A potentially serious infectious disease that mainly affects the lungs.",
    "typhoid": "A bacterial infection due to Salmonella typhi, usually spread through contaminated food/water.",
    "urinary tract infection": "An infection in any part of your urinary system (kidneys, bladder, urethra).",
    "varicose veins": "Swollen, twisted veins that lie just under the skin and usually occur in the legs.",
    "hepatitis a": "A highly contagious liver infection caused by the hepatitis A virus.",
    "paroymsal  positional vertigo": "A sudden sensation of spinning caused by changes in head position."
}

# Custom CSS
st.markdown("""
    <style>
    h1, h2, h3 {
        font-size: 26px !important;
    }
    .stMultiSelect>div>div>div>div {
        font-size: 18px;
    }
    .stSelectbox>div>div>div {
        font-size: 18px;
    }
    .stButton>button {
        font-size: 18px;
        padding: 0.4em 1.5em;
    }
    .stAlert>div {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color:#2E86C1;'>🤖 AI-Powered Disease Predictor</h1>", unsafe_allow_html=True)

# UI Inputs
st.markdown("### 🧬 Select Symptoms")
valid_symptoms = list(mlb.classes_)
user_symptoms = st.multiselect("🩺 Symptoms", options=valid_symptoms)

age_group = st.selectbox("👤 Select Age Group:", ["< 18", "18 - 35", "36 - 60", "60+"])
gender = st.selectbox("⚧ Select Gender:", ["Male", "Female", "Other"])

# Risk Logic
def get_risk_level(num_symptoms):
    if num_symptoms >= 6:
        return "High"
    elif num_symptoms >= 3:
        return "Moderate"
    else:
        return "Low"

# -------------------- Step 11: Prediction --------------------
if st.button("🔍 Predict Disease"):
    if user_symptoms:
        disease, confidence = predict_disease(user_symptoms)
        precautions = get_precautions(disease)
        description = disease_descriptions.get(disease.lower(), "No description available.")
        risk = get_risk_level(len(user_symptoms))

        st.markdown(f"<h3 style='color:green;'>🧠 Predicted Disease: <b>{disease}</b></h3>", unsafe_allow_html=True)
        st.progress(confidence / 100.0)
        st.markdown(f"<div style='font-size:18px;'>📈 Confidence: <b>{confidence}%</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px;'>🔶 Risk Level: <b>{risk}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px;'>👤 Age Group: <b>{age_group}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px;'>⚧ Gender: <b>{gender}</b></div>", unsafe_allow_html=True)

        with st.expander("📘 Disease Description"):
            st.write(description)

        with st.expander("⚠️ Recommended Precautions"):
            for i, p in enumerate(precautions, 1):
                st.markdown(f"{i}. {p}")
    else:
        st.warning("⚠️ Please select at least one symptom.")

# -------------------- Step 12: Disease Frequency Chart --------------------
st.markdown("### 📊 Top 15 Disease Occurrences in Dataset")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=symptom_df, y="Disease", order=symptom_df["Disease"].value_counts().index[:15], palette="coolwarm", ax=ax)
ax.set_title("Disease Frequency", fontsize=16)
st.pyplot(fig)

# -------------------- Step 13: Footer --------------------
st.markdown("""
---
<div style='font-size:16px;'>
Developed with ❤️ using <b>Streamlit</b>  
<br>Author: <b>Your Name</b> | Research Intern @ <b>SPSU</b>
</div>
""", unsafe_allow_html=True)

