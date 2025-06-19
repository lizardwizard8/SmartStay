# preprocess.py - functions to load and preprocess hotel guest data

import pandas as pd
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parents[1]   # <Bitirme-Projesi>
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)    

DATA_PATH = "final_synced_main_guest_names_dataset.xlsx"

# Specify numeric and categorical columns for the pipeline
_NUM_COLS = ["Guests Count", "Entrance_Hour"]
_CAT_COLS = ["Guest Gender", "Guest Country", "Room Type"]

def add_minibar_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert minibar text entries into one-hot encoded columns.
    """
    if "Items Taken from Minibar" not in df.columns:
        return df

    # Split each cell into a list of items
    lists = (
        df["Items Taken from Minibar"]
          .fillna("")
          .apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    )

    # Apply multi-label binarization
    mlb = MultiLabelBinarizer()
    ohe = pd.DataFrame(
        mlb.fit_transform(lists),
        columns=[f"Minibar_{item}" for item in mlb.classes_],
        index=df.index,
    )

    # Remove the original column and add encoded columns
    df = df.drop(columns=["Items Taken from Minibar"])
    return pd.concat([df, ohe], axis=1)

def load_raw() -> pd.DataFrame:
    """
    Read the raw dataset, extract the hour from Entrance Time,
    and encode minibar items.
    """
    # Read data from Excel
    df = pd.read_excel(DATA_PATH)

    # Convert Entrance Time to datetime and extract hour
    df["Entrance Time"] = pd.to_datetime(df["Entrance Time"], errors="coerce")
    df["Entrance_Hour"] = df["Entrance Time"].dt.hour

    # Remove the original timestamp column
    df = df.drop(columns=["Entrance Time"])

    # One-hot encode minibar selections
    df = add_minibar_ohe(df)
    return df

def make_pipeline() -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numeric scaling and one-hot encoding.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), _CAT_COLS),
        ],
        remainder="drop",
    )

def fit_transform(df: pd.DataFrame):
    """
    Fit the preprocessing pipeline on the DataFrame, save it,
    and return the transformed data and the pipeline object.
    """
    pipe = make_pipeline()
    X = pipe.fit_transform(df)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "preprocess.pkl") 
    return X, pipe

def transform_new(record):
    """
    Apply the saved pipeline to a new record (dict or DataFrame).
    """
    pipe = joblib.load(MODEL_DIR / "preprocess.pkl")
    import pandas as pd

    # Convert dict input into a single-row DataFrame
    if isinstance(record, dict):
        df = pd.DataFrame([record])
    else:
        df = record.copy()

    # If timestamp column exists, extract hour and remove it
    if "Entrance Time" in df.columns:
        df["Entrance Time"] = pd.to_datetime(df["Entrance Time"], errors="coerce")
        df["Entrance_Hour"] = df["Entrance Time"].dt.hour
        df = df.drop(columns=["Entrance Time"])

    # Return preprocessed features for the record
    return pipe.transform(df)
