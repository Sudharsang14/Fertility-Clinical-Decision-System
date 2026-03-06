import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


class SimilarityEngine:

    def __init__(self, path):

        self.df = pd.read_csv(path)

        self.features = [
            "Age","BMI","AMH","FSH","LH",
            "Diagnosis","PreviousFailures"
        ]

        # ---------- encode diagnosis ----------
        self.le = LabelEncoder()
        self.df["Diagnosis"] = self.le.fit_transform(self.df["Diagnosis"])

        # ---------- scale ----------
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(self.df[self.features])

        # ---------- knn ----------
        self.knn = NearestNeighbors(n_neighbors=5)
        self.knn.fit(X)


    # =====================================================
    # Find similar patients
    # =====================================================
    def find(self, patient):

        p = pd.DataFrame([patient])
        p["Diagnosis"] = self.le.transform([patient["Diagnosis"]])

        p_scaled = self.scaler.transform(p[self.features])

        _, idx = self.knn.kneighbors(p_scaled)

        return self.df.iloc[idx[0]]


    # =====================================================
    # ⭐ NEW: Treatment success stats
    # =====================================================
    def treatment_stats(self, patient):

        similar = self.find(patient)

        stats = []

        for treatment, group in similar.groupby("Treatment"):

            total = len(group)
            success = group["Success"].sum()

            rate = round((success / total) * 100, 2)

            stats.append({
                "Treatment": treatment,
                "Cases": total,
                "Successes": int(success),
                "SuccessRate(%)": rate
            })

        stats_df = pd.DataFrame(stats)

        stats_df = stats_df.sort_values(
            "SuccessRate(%)",
            ascending=False
        )

        return stats_df