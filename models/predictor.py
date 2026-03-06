import joblib
import pandas as pd
import shap
import numpy as np

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load("models/saved_model.pkl")


# =====================================================
# FEATURE ORDER (MUST MATCH TRAINING)
# =====================================================
FEATURE_ORDER = [
    "Age", "BMI", "AMH", "FSH", "LH",
    "Diagnosis", "PreviousFailures"
]


# =====================================================
# ENCODING
# =====================================================
DIAG_MAP = {
    "Normal": 0,
    "PCOS": 1,
    "Endometriosis": 2,
    "MaleFactor": 3,
    "Tubal": 4
}


# =====================================================
# PREPROCESS INPUT
# =====================================================
def preprocess(patient):
    p = patient.copy()
    p["Diagnosis"] = DIAG_MAP[p["Diagnosis"]]

    df = pd.DataFrame([p])
    return df[FEATURE_ORDER]


# =====================================================
# SAFE SHAP EXTRACTION FUNCTION  ⭐ KEY FIX
# =====================================================
def safe_shap_values(model, df):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    # Case 1: list (old shap)
    if isinstance(shap_values, list):
        shap_vals = shap_values[-1][0]

    # Case 2: numpy array
    else:
        shap_vals = shap_values[0]

    values = []

    for v in shap_vals:
        if isinstance(v, (list, np.ndarray)):
            values.append(float(v[0]))
        else:
            values.append(float(v))

    return values


# =====================================================
# MAIN PREDICT FUNCTION
# =====================================================
def predict_success(patient):

    df = preprocess(patient)

    # ---- probability ----
    prob = float(model.predict_proba(df)[0][1]) * 100

    # ---- shap explanation ----
    shap_vals = safe_shap_values(model, df)

    explanation = {
        FEATURE_ORDER[i]: shap_vals[i]
        for i in range(len(FEATURE_ORDER))
    }

    return round(prob, 2), explanation


# =====================================================
# TEXTUAL CLINICAL EXPLANATION
# =====================================================
def clinical_text_explanation(explanation):
    msgs = []

    for f, v in explanation.items():
        if f == "AMH":
            msgs.append(
                "Good ovarian reserve improves success" if v >= 1 else
                "Low ovarian reserve may reduce success"
            )
        elif f == "Age":
            msgs.append(
                "Younger age favors better outcome" if v < 35 else
                "Advanced age lowers implantation probability"
            )
        elif f == "BMI":
            msgs.append(
                "Healthy BMI supports treatment success" if 18.5 <= v <= 24.9 else
                "High or low BMI negatively affects fertility outcome"
            )
        elif f == "FSH":
            msgs.append(
                "Balanced FSH supports ovarian response" if v < 10 else
                "High FSH indicates reduced ovarian reserve"
            )
        elif f == "LH":
            msgs.append(
                "Balanced LH improves ovulation quality" if 5 <= v <= 20 else
                "Hormonal imbalance may affect ovulation"
            )
        elif f == "PreviousFailures":
            msgs.append(
                "Fewer past failures improve prognosis" if v == 0 else
                "Multiple failures slightly reduce success"
            )
        elif f == "Diagnosis":
            diag = str(v).lower()
            if diag == "pcos":
                msgs.append("PCOS may require specialized treatment")
            elif diag == "endometriosis":
                msgs.append("Endometriosis can affect implantation")
            else:
                msgs.append("Diagnosis type influences prognosis")

    return msgs
