
def rule_recommendation(p):
    age = p["Age"]
    amh = p["AMH"]
    diagnosis = p["Diagnosis"]

    if diagnosis == "MaleFactor":
        return "ICSI recommended (male factor infertility)"
    if diagnosis == "Tubal":
        return "IVF recommended (tubal blockage)"
    if age >= 38 or amh < 1:
        return "IVF preferred (low ovarian reserve/advanced age)"
    if diagnosis == "PCOS":
        return "Antagonist stimulation protocol"
    return "IUI or IVF based on clinician decision"
