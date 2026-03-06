import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/fertility_clinical_dataset_1200.csv")

le = LabelEncoder()
df["Diagnosis"] = le.fit_transform(df["Diagnosis"])


X = df.drop(["Success", "Treatment"], axis=1)
y = df["Success"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "models/saved_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("Model retrained successfully")