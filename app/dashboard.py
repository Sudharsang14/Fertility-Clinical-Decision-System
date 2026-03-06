import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

from rules.clinical_rules import rule_recommendation
from models.similarity_engine import SimilarityEngine
from models.predictor import predict_success, clinical_text_explanation


# =====================================================
# PATHS
# =====================================================

DATA = "data/fertility_clinical_dataset_1200.csv"
LOG_FILE = "data/prediction_log.csv"
RECORDS = "records"

os.makedirs(RECORDS, exist_ok=True)

sim_engine = SimilarityEngine(DATA)

st.set_page_config(
    page_title="Fertility Clinical Decision System",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None


# =====================================================
# HEADER NAVIGATION
# =====================================================

col1, col2 = st.columns([4,2])

with col1:
    st.markdown("## 🩺 Fertility Clinical Decision System")

with col2:
    page = st.radio(
        "",
        ["Home","Prediction","Patient History"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.divider()


# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.title("Fertility Clinical Decision Support System")

    st.write("""
This AI-powered Clinical Decision Support System assists fertility specialists
in evaluating infertility cases and recommending appropriate IVF treatment
strategies using patient clinical data.

The system integrates:

• Clinical rule-based medical guidelines  
• Machine learning success prediction  
• Similar patient outcome analysis  
• Evidence-based treatment ranking
""")

    st.success("Navigate to Prediction to analyze a patient.")

    st.divider()

    st.subheader("Understanding Fertility Treatment")

    st.write("""
Infertility affects millions of couples worldwide. Assisted Reproductive
Technologies (ART) such as **IVF (In-Vitro Fertilization)** help improve the
chances of pregnancy by combining medical expertise with advanced laboratory
procedures.

Doctors consider several factors before recommending treatment:

• Patient age  
• Body Mass Index (BMI)  
• Hormone levels (AMH, FSH, LH)  
• Previous treatment failures  
• Diagnosis such as PCOS, Endometriosis, or Tubal factors
""")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            "https://images.unsplash.com/photo-1581594693702-fbdc51b2763b",
            caption="Fertility Consultation",
            use_container_width=True
        )

    with col2:
        st.image(
            "https://images.unsplash.com/photo-1579154204601-01588f351e67",
            caption="IVF Laboratory Process",
            use_container_width=True
        )

    with col3:
        st.image(
            "https://images.unsplash.com/photo-1579684385127-1ef15d508118",
            caption="Embryo Development",
            use_container_width=True
        )


# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.title("🔬 Patient Fertility Prediction")

    with st.form("patient_form"):

        st.subheader("Enter Patient Details")

        patient_name = st.text_input("Patient Name *")

        col1,col2 = st.columns(2)

        with col1:
            age = st.number_input("Age",18,50,30)
            bmi = st.number_input("BMI",10.0,50.0,24.0)
            amh = st.number_input("AMH",0.0,10.0,2.5)

        with col2:
            fsh = st.number_input("FSH",0.0,30.0,7.0)
            lh = st.number_input("LH",0.0,30.0,6.0)
            fails = st.number_input("Previous Failures",0,10,0)

        diag = st.selectbox(
            "Diagnosis",
            ["Normal","PCOS","Endometriosis","MaleFactor","Tubal"]
        )

        submit = st.form_submit_button("Analyze")


    if submit:

        if patient_name.strip()=="":
            st.error("⚠ Please enter patient name")
            st.stop()

        st.session_state.analysis_done = True

        st.session_state.patient_data = {
            "PatientName":patient_name,
            "Age":age,
            "BMI":bmi,
            "AMH":amh,
            "FSH":fsh,
            "LH":lh,
            "Diagnosis":diag,
            "PreviousFailures":fails
        }


    # =====================================================
    # RUN ANALYSIS
    # =====================================================

    if st.session_state.analysis_done:

        pdata = st.session_state.patient_data

        patient={
            "Age":pdata["Age"],
            "BMI":pdata["BMI"],
            "AMH":pdata["AMH"],
            "FSH":pdata["FSH"],
            "LH":pdata["LH"],
            "Diagnosis":pdata["Diagnosis"],
            "PreviousFailures":pdata["PreviousFailures"]
        }

        patient_name = pdata["PatientName"]

        st.divider()

        # =====================================================
        # RULE ENGINE
        # =====================================================

        st.subheader("📋 Rule-based Recommendation")

        rule = rule_recommendation(patient)
        st.success(rule)


        # =====================================================
        # ML PREDICTION
        # =====================================================

        st.subheader("📈 ML Success Prediction")

        prob, explanation = predict_success(patient)

        st.metric(
            label="Predicted IVF Success Probability",
            value=f"{prob}%"
        )


        # =====================================================
        # FEATURE IMPACT
        # =====================================================

        st.subheader("🔍 Feature Impact")

        exp_df = pd.DataFrame({
            "Feature":list(explanation.keys()),
            "Impact":list(explanation.values())
        })

        st.dataframe(exp_df)


        # =====================================================
        # IMPACT CHART
        # =====================================================
        st.subheader("📊 Impact Visualization")

        colors = ["green" if i>0 else "red" for i in exp_df["Impact"]]

        fig = plt.figure(figsize=(4,3.5))

        plt.barh(
            exp_df["Feature"],
            exp_df["Impact"],
            height=0.35,
            color=colors
        )

        plt.axvline(0,color="black")
        plt.gca().invert_yaxis()

        plt.tight_layout()

        st.pyplot(fig,use_container_width=False)
        
        # =====================================================
        # CLINICAL EXPLANATION
        # =====================================================

        st.subheader("🧠 Clinical Interpretation")

        messages = clinical_text_explanation(explanation)

        for msg in messages:
            st.write("•",msg)

        # =====================================================
        # SIMILAR PATIENTS
        # =====================================================

        st.subheader("👥 Similar Patients")

        similar = sim_engine.find(patient)

        st.dataframe(similar)


        # =====================================================
        # TREATMENT SUMMARY
        # =====================================================

        st.subheader("📊 Treatment Success Summary")

        stats = sim_engine.treatment_stats(patient)

        st.dataframe(stats)
        st.subheader("📈 Treatment Success Ranking")

        fig2 = plt.figure(figsize=(4,3.5))

        plt.bar(
            stats["Treatment"],
            stats["SuccessRate(%)"]
        )

        plt.ylabel("Success Rate (%)")

        plt.tight_layout()

        st.pyplot(fig2,use_container_width=False)

        # =====================================================
        # AI RECOMMENDATION
        # =====================================================

        st.subheader("🏆 AI Recommended Treatment")

        best = stats.iloc[0]

        st.success(
f"""
AI Recommendation: {best['Treatment']}

Success Rate: {best['SuccessRate(%)']}%
"""
        )


        # =====================================================
        # DOCTOR OVERRIDE
        # =====================================================

        st.subheader("👨‍⚕️ Doctor Override")

        override = st.checkbox("Doctor wants to override AI recommendation")

        override_reason = ""

        # FULL TREATMENT LIST (NOT DEPENDENT ON STATS)
        treatment_options = [
            "IVF",
            "ICSI",
            "IUI",
            "Frozen Embryo Transfer",
            "Ovulation Induction",
            "Donor Egg IVF",
            "Donor Sperm IUI",
            "Embryo Donation"
        ]

        if override:

            best_treatment = st.selectbox(
                "Select Alternative Treatment",
                treatment_options
            )

            override_reason = st.text_area(
                "Reason for Override"
            )

            st.warning(f"Doctor selected treatment: {best_treatment}")

        else:

            best_treatment = best["Treatment"]


        # =====================================================
        # SAVE DECISION
        # =====================================================

        if st.button("Save Decision"):

            row={
                "PatientName":patient_name,
                "Age":pdata["Age"],
                "BMI":pdata["BMI"],
                "Diagnosis":pdata["Diagnosis"],
                "PredictedSuccess":prob,
                "FinalTreatment":best_treatment,
                "OverrideReason":override_reason
            }

            df = pd.DataFrame([row])

            if os.path.exists(LOG_FILE):
                df.to_csv(LOG_FILE,mode="a",header=False,index=False)
            else:
                df.to_csv(LOG_FILE,index=False)

            st.success("Decision saved")


        # =====================================================
        # PDF REPORT
        # =====================================================

        if st.button("Generate PDF Report"):

            pdf = FPDF()
            pdf.add_page()

            pdf.set_font("Arial","B",16)
            pdf.cell(0,10,"Fertility Clinical Decision Report",ln=True)

            pdf.set_font("Arial","",12)

            pdf.cell(0,10,f"Patient Name: {patient_name}",ln=True)
            pdf.cell(0,10,f"Age: {pdata['Age']}",ln=True)
            pdf.cell(0,10,f"Diagnosis: {pdata['Diagnosis']}",ln=True)
            pdf.cell(0,10,f"Predicted Success: {prob}%",ln=True)

            pdf.cell(0,10,f"AI Recommendation: {best['Treatment']}",ln=True)

            pdf.cell(0,10,f"Final Treatment: {best_treatment}",ln=True)

            if override:
                pdf.multi_cell(0,10,f"Override Reason: {override_reason}")

            pdf_path = f"{RECORDS}/{patient_name}.pdf"

            pdf.output(pdf_path)

            with open(pdf_path,"rb") as f:

                st.download_button(
                    "Download Patient Report (PDF)",
                    f,
                    file_name=f"{patient_name}_report.pdf"
                )


# =====================================================
# HISTORY PAGE
# =====================================================

elif page == "Patient History":

    st.title("📜 Patient Prediction History")

    if os.path.exists(LOG_FILE):

        history = pd.read_csv(LOG_FILE)

        history["PatientName"] = history["PatientName"].fillna("")

        search = st.text_input("Search Patient")

        if search:

            history = history[
                history["PatientName"].str.contains(search,case=False,na=False)
            ]

        st.dataframe(history)

    else:

        st.info("No history available yet.")
