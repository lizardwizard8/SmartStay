import os
import joblib
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from preprocess import load_raw, fit_transform

# ----------------------------------------------------------------------
# 0. Regular expression utilities for parsing In-Out timestamps
# ----------------------------------------------------------------------
# Compile a regex pattern to find pairs of Out:HH:MM and In:HH:MM in text
_OUT_IN_RE = re.compile(
    r"Out:\s*(\d{2}:\d{2}(?::\d{2})?)\s*,?\s*In:\s*(\d{2}:\d{2}(?::\d{2})?)",
    flags=re.I
)

def _time_to_min(t: str) -> float:
    """
    Convert a time string "HH:MM" or "HH:MM:SS" into minutes since midnight.
    Splits on ':' and accounts for optional seconds. Returns hours*60 + minutes + seconds/60.
    """
    parts = t.split(":")
    h = int(parts[0])                # hours component
    m = int(parts[1])                # minutes component
    s = int(parts[2]) if len(parts) == 3 else 0  # optional seconds
    return h * 60 + m + s / 60       # total minutes (with fractional part)


def longest_empty_window(text: str):
    """
    From a cell containing multiple "Out/In" pairs, find the largest gap.
    Returns the start and end times (in minutes) of the longest period when the room was empty.
    If no valid data, returns (None, None).
    """
    if pd.isna(text):                # if the cell is NaN, no data
        return None, None
    best_gap = -1                    # track the largest duration found
    best_start = best_end = None
    # Find all matching (out_time, in_time) pairs
    for out_t, in_t in _OUT_IN_RE.findall(text):
        start = _time_to_min(out_t)  # convert 'Out' timestamp to minutes
        end   = _time_to_min(in_t)   # convert 'In' timestamp to minutes
        if end < start:
            end += 24 * 60          # account for crossings past midnight
        gap = end - start           # duration in minutes
        # Update best if this gap is longer
        if gap > best_gap:
            best_gap, best_start, best_end = gap, start, end
    return best_start, best_end      # return minutes for the longest interval


def last_seen_time(text: str) -> float:
    """
    From an "Out/In" log, return the last 'In' time in minutes since midnight.
    If no valid times, returns NaN.
    """
    if pd.isna(text):                # handle missing data
        return np.nan
    pairs = _OUT_IN_RE.findall(text)
    if not pairs:                    # no matches found
        return np.nan
    # Take the last match's 'In' timestamp
    _, in_t = pairs[-1]
    return _time_to_min(in_t)

# ----------------------------------------------------------------------
# 1. Load data and preprocessing pipeline
# ----------------------------------------------------------------------
os.makedirs("models", exist_ok=True)   # ensure model directory exists
RAW = load_raw()                         # load raw DataFrame with minibar flags and entrance hour
X_sp, pipe = fit_transform(RAW)          # sparse feature matrix and saved pipeline

# Load the precomputed KMeans clustering model and assign cluster IDs
kmeans = joblib.load("models/kmeans.pkl")
cluster_id = kmeans.predict(pipe.transform(RAW)).reshape(-1, 1)
# Combine feature matrix and cluster ID into a full array for model training
X = np.hstack([X_sp.toarray(), cluster_id])  # shape: (num_samples, num_features+1)

# ----------------------------------------------------------------------
# 2. Compute empty-room time targets
# ----------------------------------------------------------------------
empty_start = []  # list to collect start times of the longest empty window
empty_end   = []  # list to collect end times
for txt in RAW.get("In-Out Times", []):
    s, e = longest_empty_window(txt)
    empty_start.append(s)
    empty_end.append(e)
# Add new columns to RAW in hours (minutes/60)
RAW["Empty_Start_Hour"] = np.array(empty_start) / 60.0
RAW["Empty_End_Hour"]   = np.array(empty_end)   / 60.0

# ----------------------------------------------------------------------
# 3. Define prediction targets
# ----------------------------------------------------------------------
y_temp   = RAW["Preferred Heat (°C)"].values
y_dimmer = RAW["Preferred Dimmer Level"].values
y_start  = RAW["Empty_Start_Hour"].values
y_end    = RAW["Empty_End_Hour"].values
# Minibar one-hot columns start with 'Minibar_'
minibar_cols = [c for c in RAW.columns if c.startswith("Minibar_")]
if not minibar_cols:
    raise ValueError("No minibar columns found; check your preprocessing step!")
Y_minibar = RAW[minibar_cols]            # DataFrame of binary minibar flags

# ----------------------------------------------------------------------
# 4. Encode room direction for classification
# ----------------------------------------------------------------------
raw_dirs = RAW["Room Direction"].astype(str).values
le_dir = LabelEncoder()                  # create label encoder
y_dir = le_dir.fit_transform(raw_dirs)   # integer codes for each direction
joblib.dump(le_dir, "models/dir_encoder.pkl")  # save encoder for later inverse-transform

# ----------------------------------------------------------------------
# 5. Initialize regression and classification models
# ----------------------------------------------------------------------
# Temperature preference regressor: limits depth and ensemble size for generalization
temp_model      = RandomForestRegressor(max_depth=6, n_estimators=200, random_state=0)
# Dimmer preference: simple linear regression
dimmer_model    = LinearRegression()
# Minibar sequence predictor: multi-output boosting classifier for each item flag
minibar_model   = MultiOutputClassifier(GradientBoostingClassifier(random_state=42))
# Models for empty-room window start and end times
empty_start_mod = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
empty_end_mod   = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
# Room direction classifier: random forest for multi-class direction prediction
dir_model       = RandomForestClassifier(n_estimators=100, random_state=42)

# ----------------------------------------------------------------------
# 6. Train all models on full dataset
# ----------------------------------------------------------------------
temp_model.fit(X, y_temp)                # learn temperature mapping
dimmer_model.fit(X, y_dimmer)            # learn dimmer level mapping
minibar_model.fit(X, Y_minibar)          # learn minibar item presence patterns
empty_start_mod.fit(X, y_start)          # learn start times for empty windows
empty_end_mod.fit(X, y_end)              # learn end times for empty windows
dir_model.fit(X, y_dir)                  # learn room direction classification

# ----------------------------------------------------------------------
# 7. Save trained model artifacts for inference
# ----------------------------------------------------------------------
joblib.dump(temp_model,      "models/temp_reg.pkl")
joblib.dump(dimmer_model,    "models/dimmer_reg.pkl")
joblib.dump(minibar_model,   "models/minibar_model.pkl")
joblib.dump(empty_start_mod, "models/empty_start_reg.pkl")
joblib.dump(empty_end_mod,   "models/empty_end_reg.pkl")
joblib.dump(dir_model,       "models/dir_model.pkl")

# ----------------------------------------------------------------------
# 8. Quick training diagnostics: report R² and MAE for regressors, accuracy for classifier
# ----------------------------------------------------------------------
def reg_report(name: str, y_true, y_pred):
    """
    Print a summary of regression performance metrics: R² and mean absolute error.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"=== {name} ===   R²: {r2:.3f}   MAE: {mae:.2f}")

# Display performance for each regression target
reg_report("Temperature",       y_temp,   temp_model.predict(X))
reg_report("Dimmer",            y_dimmer, dimmer_model.predict(X))
reg_report("Empty-start hour",  y_start,  empty_start_mod.predict(X))
reg_report("Empty-end hour",    y_end,    empty_end_mod.predict(X))

# Classification accuracy for room direction
accuracy_dir = (dir_model.predict(X) == y_dir).mean()
print(f"=== Room-Direction Classification accuracy: {accuracy_dir:.3f}")

if __name__ == "__main__":
    print("\n✅ Training complete — model files saved under ./models/")
