"""
app.py  –  Disease Predictor & Risk Analysis
A multi-disease prediction web app built with Streamlit.
Predicts: Diabetes | Heart Disease | Liver Disease
Model: Logistic Regression (scikit-learn)
"""

import streamlit as st
import numpy as np
import pickle
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Predictor & Risk Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

    .main { background: #0d1117; }
    .block-container { padding-top: 2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    [data-testid="stSidebar"] * { color: #e0f4ff !important; }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2a38 0%, #243447 100%);
        border: 1px solid #2d4a6e;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .result-safe {
        background: linear-gradient(135deg, #0d2f1e, #1a4a2e);
        border: 2px solid #2ecc71;
        border-radius: 16px;
        padding: 1.5rem;
        color: #2ecc71;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
    }
    .result-risk {
        background: linear-gradient(135deg, #2f0d0d, #4a1a1a);
        border: 2px solid #e74c3c;
        border-radius: 16px;
        padding: 1.5rem;
        color: #e74c3c;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
    }
    .info-box {
        background: #1a2433;
        border-left: 4px solid #3498db;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #a8c7e8;
        font-size: 0.9rem;
    }
    .section-header {
        color: #5dade2;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1.2rem 0 0.6rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(21,101,192,0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1976d2, #1565c0);
        box-shadow: 0 6px 20px rgba(21,101,192,0.6);
        transform: translateY(-2px);
    }
    .risk-bar-container {
        background: #1e2a38;
        border-radius: 50px;
        height: 20px;
        overflow: hidden;
        margin: 0.3rem 0;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label { color: #8ab4d4 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    models = {}
    for name in ["diabetes", "heart", "liver"]:
        path = os.path.join(base, "models", f"{name}_model.pkl")
        if os.path.exists(path):
            models[name] = pickle.load(open(path, "rb"))
    return models

models = load_models()


# ─── Helpers ───────────────────────────────────────────────────────────────────
def risk_color(prob):
    if prob < 0.35:   return "#2ecc71"
    elif prob < 0.60: return "#f39c12"
    else:             return "#e74c3c"

def risk_label(prob):
    if prob < 0.35:   return "🟢 LOW RISK"
    elif prob < 0.60: return "🟡 MODERATE RISK"
    else:             return "🔴 HIGH RISK"

def render_risk_bar(prob):
    color = risk_color(prob)
    pct = int(prob * 100)
    st.markdown(f"""
    <div class='risk-bar-container'>
        <div style='width:{pct}%; height:100%; background:{color};
                    border-radius:50px; transition:width 1s ease;'></div>
    </div>
    <p style='color:{color}; font-weight:600; font-size:0.95rem; margin:0.2rem 0;'>
        Risk Score: {pct}% — {risk_label(prob)}
    </p>
    """, unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Disease Predictor")
    st.markdown("**Final Year CS Project**")
    st.markdown("---")
    disease = st.radio(
        "Select Disease to Predict:",
        ["🩸 Diabetes", "❤️ Heart Disease", "🫀 Liver Disease", "📊 Full Health Report"],
        index=0,
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#7fb3d3;'>
    <b>About this App</b><br><br>
    Uses <b>Logistic Regression</b> to predict disease risk based on clinical parameters.<br><br>
    ⚠️ <i>For educational purposes only. Always consult a doctor.</i>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DIABETES
# ═══════════════════════════════════════════════════════════════════════════════
if disease == "🩸 Diabetes":
    st.markdown("# 🩸 Diabetes Risk Prediction")
    st.markdown("<div class='info-box'>Enter the patient's clinical values below. All fields are required.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='section-header'>Personal Info</div>", unsafe_allow_html=True)
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1)
        age = st.number_input("Age (years)", 18, 100, 30)
        dpf = st.number_input("Diabetes Pedigree Function", 0.05, 3.0, 0.5, step=0.01,
                              help="A function that scores likelihood of diabetes based on family history. Typical: 0.08–2.5")

    with col2:
        st.markdown("<div class='section-header'>Blood Parameters</div>", unsafe_allow_html=True)
        glucose = st.number_input("Glucose Level (mg/dL)", 50, 250, 110,
                                  help="Plasma glucose concentration (2-hour oral glucose tolerance test)")
        insulin = st.number_input("Insulin (mu U/ml)", 0, 600, 80)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 40, 130, 72)

    with col3:
        st.markdown("<div class='section-header'>Body Measurements</div>", unsafe_allow_html=True)
        bmi = st.number_input("BMI (kg/m²)", 15.0, 70.0, 28.0, step=0.1)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 25)

    st.markdown("")
    if st.button("🔍 Predict Diabetes Risk"):
        model, scaler = models["diabetes"]
        features = np.array([[pregnancies, glucose, blood_pressure,
                              skin_thickness, insulin, bmi, dpf, age]])
        scaled = scaler.transform(features)
        prob = model.predict_proba(scaled)[0][1]
        pred = model.predict(scaled)[0]

        st.markdown("---")
        st.markdown("## 📋 Prediction Result")
        c1, c2 = st.columns([1, 1])
        with c1:
            if pred == 1:
                st.markdown("<div class='result-risk'>⚠️ DIABETES DETECTED<br><small>High probability of diabetes</small></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-safe'>✅ NO DIABETES DETECTED<br><small>Low probability of diabetes</small></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("**Risk Analysis**")
            render_risk_bar(prob)
            st.markdown("</div>", unsafe_allow_html=True)

        # Key factors
        st.markdown("### 💡 Key Risk Factors Analysis")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Glucose", f"{glucose} mg/dL", delta="High" if glucose > 140 else "Normal")
        f2.metric("BMI", f"{bmi}", delta="Obese" if bmi > 30 else "Normal")
        f3.metric("Blood Pressure", f"{blood_pressure} mmHg", delta="High" if blood_pressure > 90 else "Normal")
        f4.metric("Age", f"{age} yrs", delta="Risk Factor" if age > 45 else "OK")

        st.markdown("<div class='info-box'>⚕️ <b>Disclaimer:</b> This prediction is based on a machine learning model trained on sample data. Always consult a certified medical professional for accurate diagnosis.</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HEART DISEASE
# ═══════════════════════════════════════════════════════════════════════════════
elif disease == "❤️ Heart Disease":
    st.markdown("# ❤️ Heart Disease Risk Prediction")
    st.markdown("<div class='info-box'>Enter the patient's clinical and diagnostic values below.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='section-header'>Personal Info</div>", unsafe_allow_html=True)
        age = st.number_input("Age (years)", 20, 100, 45)
        sex = st.selectbox("Sex", ["Female", "Male"])
        sex_val = 1 if sex == "Male" else 0
        cp = st.selectbox("Chest Pain Type",
                          ["0 – Typical Angina", "1 – Atypical Angina",
                           "2 – Non-Anginal Pain", "3 – Asymptomatic"])
        cp_val = int(cp[0])

    with col2:
        st.markdown("<div class='section-header'>Vitals & Lab Values</div>", unsafe_allow_html=True)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 210, 130)
        chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No (0)", "Yes (1)"])
        fbs_val = int(fbs[0] == "Y")
        restecg = st.selectbox("Resting ECG Result",
                               ["0 – Normal", "1 – ST-T Wave Abnormality", "2 – LV Hypertrophy"])
        restecg_val = int(restecg[0])

    with col3:
        st.markdown("<div class='section-header'>Exercise Test</div>", unsafe_allow_html=True)
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina?", ["No (0)", "Yes (1)"])
        exang_val = 1 if "Yes" in exang else 0
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak ST Segment",
                             ["0 – Upsloping", "1 – Flat", "2 – Downsloping"])
        slope_val = int(slope[0])
        ca = st.number_input("Major Vessels Coloured by Fluoroscopy (0–3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", ["0 – Normal", "1 – Fixed Defect", "2 – Reversible Defect"])
        thal_val = int(thal[0])

    st.markdown("")
    if st.button("🔍 Predict Heart Disease Risk"):
        model, scaler = models["heart"]
        features = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                              restecg_val, thalach, exang_val, oldpeak,
                              slope_val, ca, thal_val]])
        scaled = scaler.transform(features)
        prob = model.predict_proba(scaled)[0][1]
        pred = model.predict(scaled)[0]

        st.markdown("---")
        st.markdown("## 📋 Prediction Result")
        c1, c2 = st.columns([1, 1])
        with c1:
            if pred == 1:
                st.markdown("<div class='result-risk'>⚠️ HEART DISEASE DETECTED<br><small>High probability of heart disease</small></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-safe'>✅ NO HEART DISEASE DETECTED<br><small>Low probability of heart disease</small></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("**Risk Analysis**")
            render_risk_bar(prob)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 💡 Key Risk Factors Analysis")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Cholesterol", f"{chol} mg/dL", delta="High" if chol > 240 else "Normal")
        f2.metric("Blood Pressure", f"{trestbps} mmHg", delta="High" if trestbps > 140 else "Normal")
        f3.metric("Max Heart Rate", f"{thalach}", delta="Low" if thalach < 100 else "Normal")
        f4.metric("Chest Pain Type", cp[0], delta="Risky" if cp_val == 3 else "OK")

        st.markdown("<div class='info-box'>⚕️ <b>Disclaimer:</b> This prediction is for educational purposes only. Consult a cardiologist for any heart-related concerns.</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVER DISEASE
# ═══════════════════════════════════════════════════════════════════════════════
elif disease == "🫀 Liver Disease":
    st.markdown("# 🫀 Liver Disease Risk Prediction")
    st.markdown("<div class='info-box'>Enter liver function test (LFT) values for prediction.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='section-header'>Personal Info</div>", unsafe_allow_html=True)
        age = st.number_input("Age (years)", 4, 95, 35)
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender_val = 1 if gender == "Male" else 0

    with col2:
        st.markdown("<div class='section-header'>Bilirubin Values</div>", unsafe_allow_html=True)
        total_bili = st.number_input("Total Bilirubin (mg/dL)", 0.4, 30.0, 1.0, step=0.1,
                                     help="Normal: 0.1–1.2 mg/dL")
        direct_bili = st.number_input("Direct Bilirubin (mg/dL)", 0.1, 15.0, 0.3, step=0.1,
                                      help="Normal: 0–0.3 mg/dL")
        alk_phos = st.number_input("Alkaline Phosphotase (IU/L)", 60, 800, 210,
                                   help="Normal: 44–147 IU/L")

    with col3:
        st.markdown("<div class='section-header'>Enzyme & Protein Tests</div>", unsafe_allow_html=True)
        alamine = st.number_input("Alamine Aminotransferase – ALT (IU/L)", 7, 700, 35,
                                  help="Normal: 7–56 IU/L")
        aspartate = st.number_input("Aspartate Aminotransferase – AST (IU/L)", 10, 800, 40,
                                    help="Normal: 10–40 IU/L")
        total_proteins = st.number_input("Total Proteins (g/dL)", 2.7, 9.6, 6.5, step=0.1)
        albumin = st.number_input("Albumin (g/dL)", 0.9, 5.5, 3.3, step=0.1)
        ag_ratio = st.number_input("Albumin/Globulin Ratio", 0.3, 3.0, 1.0, step=0.1)

    st.markdown("")
    if st.button("🔍 Predict Liver Disease Risk"):
        model, scaler = models["liver"]
        features = np.array([[age, gender_val, total_bili, direct_bili,
                              alk_phos, alamine, aspartate,
                              total_proteins, albumin, ag_ratio]])
        scaled = scaler.transform(features)
        prob = model.predict_proba(scaled)[0][1]
        pred = model.predict(scaled)[0]

        st.markdown("---")
        st.markdown("## 📋 Prediction Result")
        c1, c2 = st.columns([1, 1])
        with c1:
            if pred == 1:
                st.markdown("<div class='result-risk'>⚠️ LIVER DISEASE DETECTED<br><small>High probability of liver disease</small></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-safe'>✅ NO LIVER DISEASE DETECTED<br><small>Low probability of liver disease</small></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("**Risk Analysis**")
            render_risk_bar(prob)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 💡 Key Risk Factors Analysis")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Total Bilirubin", f"{total_bili}", delta="High" if total_bili > 1.2 else "Normal")
        f2.metric("ALT", f"{alamine} IU/L", delta="High" if alamine > 56 else "Normal")
        f3.metric("AST", f"{aspartate} IU/L", delta="High" if aspartate > 40 else "Normal")
        f4.metric("Albumin", f"{albumin} g/dL", delta="Low" if albumin < 3.5 else "Normal")

        st.markdown("<div class='info-box'>⚕️ <b>Disclaimer:</b> Liver disease assessment requires multiple tests. Consult a hepatologist for accurate diagnosis.</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FULL HEALTH REPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif disease == "📊 Full Health Report":
    st.markdown("# 📊 Full Health Risk Report")
    st.markdown("<div class='info-box'>Enter basic patient information to get a combined risk analysis across all three diseases.</div>", unsafe_allow_html=True)

    st.markdown("### 👤 Patient Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["Female", "Male"])
    with c2:
        bmi = st.number_input("BMI", 15.0, 60.0, 26.0, step=0.1)
        glucose = st.number_input("Glucose (mg/dL)", 50, 250, 100)
    with c3:
        chol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
        bp = st.number_input("Blood Pressure (mmHg)", 60, 180, 120)

    if st.button("🔍 Generate Full Health Report"):
        sex_val = 1 if sex == "Male" else 0

        # Diabetes prediction (use defaults for unsupplied fields)
        dm, ds = models["diabetes"]
        d_feat = np.array([[0, glucose, bp, 20, 80, bmi, 0.5, age]])
        d_prob = dm.predict_proba(ds.transform(d_feat))[0][1]

        # Heart prediction
        hm, hs = models["heart"]
        h_feat = np.array([[age, sex_val, 1, bp, chol, 0, 0, 150, 0, 1.0, 1, 0, 1]])
        h_prob = hm.predict_proba(hs.transform(h_feat))[0][1]

        # Liver prediction
        lm, ls = models["liver"]
        l_feat = np.array([[age, sex_val, 1.0, 0.3, 200, 35, 40, 6.5, 3.3, 1.0]])
        l_prob = lm.predict_proba(ls.transform(l_feat))[0][1]

        st.markdown("---")
        st.markdown("## 📋 Overall Health Report")

        r1, r2, r3 = st.columns(3)
        for col, label, prob, icon in [
            (r1, "Diabetes", d_prob, "🩸"),
            (r2, "Heart Disease", h_prob, "❤️"),
            (r3, "Liver Disease", l_prob, "🫀"),
        ]:
            with col:
                color = risk_color(prob)
                st.markdown(f"""
                <div class='metric-card' style='border-color:{color};'>
                    <h3 style='color:{color};margin:0'>{icon} {label}</h3>
                    <h2 style='color:{color};margin:0.5rem 0'>{int(prob*100)}%</h2>
                    <p style='color:#aaa;margin:0'>{risk_label(prob)}</p>
                </div>
                """, unsafe_allow_html=True)
                render_risk_bar(prob)

        # Overall health score
        overall = (d_prob + h_prob + l_prob) / 3
        st.markdown("### 🏥 Overall Health Risk Score")
        render_risk_bar(overall)

        # Recommendations
        st.markdown("### 📌 Recommendations")
        recs = []
        if d_prob > 0.4:
            recs.append("🩸 **Diabetes:** Monitor blood sugar levels; reduce sugar and refined carb intake.")
        if h_prob > 0.4:
            recs.append("❤️ **Heart:** Reduce sodium, exercise regularly, and monitor blood pressure.")
        if l_prob > 0.4:
            recs.append("🫀 **Liver:** Limit alcohol, avoid fatty foods, and get liver function tests.")
        if not recs:
            recs.append("✅ Your risk levels are relatively low. Maintain a healthy lifestyle!")

        for r in recs:
            st.markdown(f"- {r}")

        st.markdown("<div class='info-box'>⚕️ <b>Important:</b> This full report uses default values for parameters not entered. For a precise result, use each disease tab individually. Always consult a doctor.</div>", unsafe_allow_html=True)
