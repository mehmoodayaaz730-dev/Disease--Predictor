# 🏥 Disease Predictor & Risk Analysis
### Final Year Computer Science Project
**Tech Stack:** Python · Streamlit · Scikit-learn · Logistic Regression

---

## 📋 Project Overview
A web-based multi-disease prediction system that uses **Logistic Regression** (a supervised machine learning algorithm) to predict the risk of:
- 🩸 **Diabetes** (based on blood glucose, BMI, age, etc.)
- ❤️ **Heart Disease** (based on cholesterol, ECG, chest pain type, etc.)
- 🫀 **Liver Disease** (based on liver enzyme tests, bilirubin, etc.)

---

## 🗂️ Project Structure
```
disease_predictor/
├── app.py              ← Main Streamlit web application
├── model_trainer.py    ← Trains and saves ML models
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── models/
    ├── diabetes_model.pkl
    ├── heart_model.pkl
    └── liver_model.pkl
```

---

## ⚙️ How to Run (Step-by-Step)

### Step 1 – Install Python
Make sure Python 3.8+ is installed. Download from: https://www.python.org

### Step 2 – Install Dependencies
Open terminal/command prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 3 – Train the Models
```bash
python model_trainer.py
```
This creates the `.pkl` model files inside the `models/` folder.

### Step 4 – Launch the Web App
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

---

## 🧠 How It Works (For Viva)

### Machine Learning Algorithm: Logistic Regression
- **Type:** Supervised Classification Algorithm
- **Output:** Probability (0 to 1) of having the disease
- **Why Logistic Regression?**
  - Easy to understand and explain
  - Works well for binary classification (disease / no disease)
  - Gives probability scores, not just yes/no
  - Very popular in medical ML research

### Workflow:
```
Patient Data Input
       ↓
Feature Scaling (StandardScaler)
       ↓
Logistic Regression Model
       ↓
Probability Score (0–100%)
       ↓
Risk Classification: LOW / MODERATE / HIGH
```

---

## 📊 Model Accuracy
| Disease | Accuracy |
|---------|----------|
| Diabetes | ~70% |
| Heart Disease | ~77% |
| Liver Disease | ~69% |

---

## 💡 Key Features
- Multi-disease prediction in one app
- Clean, professional web UI
- Real-time risk percentage with color-coded bar
- Metric analysis for key clinical parameters
- Full Health Report combining all three diseases
- Medical disclaimer for ethical compliance

---

## 🔬 Future Improvements (for report)
- Use real datasets (UCI, Kaggle)
- Add Random Forest / SVM for better accuracy
- Add patient history tracking with database
- Deploy on cloud (Heroku / AWS)
- Add PDF report generation

---

*⚕️ Disclaimer: For educational purposes only. Not for actual medical diagnosis.*
