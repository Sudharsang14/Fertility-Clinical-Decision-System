
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    def __init__(self):
        self.le = LabelEncoder()
        self.scaler = StandardScaler()

    def fit(self, df, cols):
        df["Diagnosis"] = self.le.fit_transform(df["Diagnosis"])
        self.scaler.fit(df[cols])

    def transform_row(self, row, cols):
        import pandas as pd
        r = pd.DataFrame([row])
        r["Diagnosis"] = self.le.transform([row["Diagnosis"]])
        return self.scaler.transform(r[cols])
