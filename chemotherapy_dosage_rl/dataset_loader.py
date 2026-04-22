import numpy as np
import pandas as pd


DATASET_PATHS = {
    "lung_carboplatin": "lung_carboplatin_cleaned.csv",
    "lung_cisplatin": "lung_cisplatin_cleaned.csv",
    "crc_folfox": "crc_folfox_cleaned.csv",
    "crc_oxali": "crc_oxali_cleaned.csv",
    "crc_5fu": "crc_5fu_cleaned.csv",
}


class DatasetPatientBuilder:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]

        self.response_col = self._find_col(["cleanresponse", "response", "bestresponse"])
        self.age_col = self._find_col(["ageclean", "age"])
        self.stage_col = self._find_col(["tumorstage", "stage"])
        self.start_col = self._find_col(["cleantherapystart"])
        self.end_col = self._find_col(["cleantherapyend"])
        self.immune_cols = [c for c in self.df.columns if "quanTIseq" in c][:6]

    def _find_col(self, candidates):
        lower = {c.lower(): c for c in self.df.columns}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    def _norm(self, value, lo, hi, default=0.5):
        if pd.isna(value):
            return default
        if hi <= lo:
            return default
        x = (float(value) - lo) / (hi - lo)
        return float(np.clip(x, 0.0, 1.0))

    def _stage_score(self, text):
        if pd.isna(text):
            return 0.5
        t = str(text).lower()
        if "iv" in t:
            return 1.0
        if "iii" in t:
            return 0.75
        if "ii" in t:
            return 0.5
        if "i" in t:
            return 0.25
        return 0.5

    def _duration_score(self, row):
        if self.start_col is None or self.end_col is None:
            return 0.5
        try:
            a = float(row[self.start_col])
            b = float(row[self.end_col])
            d = max(0.0, b - a)
            return float(np.clip(d / 365.0, 0.0, 1.0))
        except Exception:
            return 0.5

    def _immune_score(self, row):
        vals = []
        for c in self.immune_cols:
            try:
                vals.append(float(row[c]))
            except Exception:
                pass
        if not vals:
            return 0.5
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.5
        mean_val = float(np.mean(arr))
        return float(np.clip(mean_val / (np.max(arr) + 1e-8), 0.0, 1.0))

    def sample_patient(self):
        row = self.df.sample(n=1).iloc[0]
        return self.row_to_patient(row), row

    def row_to_patient(self, row):
        response = float(row[self.response_col]) if self.response_col and pd.notna(row[self.response_col]) else 0.5
        age = float(row[self.age_col]) if self.age_col and pd.notna(row[self.age_col]) else 60.0
        stage = self._stage_score(row[self.stage_col]) if self.stage_col else 0.5
        duration = self._duration_score(row)
        immune = self._immune_score(row)

        age_n = self._norm(age, 20, 90, 0.5)
        response_n = float(np.clip(response, 0.0, 1.0))

        patient = {
            "r_t": 1.0 + 0.5 * stage,
            "k_t": 1.0,
            "kill_t": 0.55 + 0.45 * response_n,
            "r_h": 0.55 + 0.25 * (1.0 - age_n),
            "k_h": 1.0,
            "kill_h": 0.20 + 0.35 * age_n + 0.10 * (1.0 - immune),
            "drug_decay": 0.7 + 0.6 * duration,
            "init_T": 0.45 + 0.45 * stage,
            "init_H": 0.75 + 0.25 * (1.0 - age_n),
            "init_C": 0.0,
            "meta_response": response_n,
            "meta_age": age,
            "meta_stage": stage,
            "meta_immune": immune,
        }
        return patient