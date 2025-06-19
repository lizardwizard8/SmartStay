# Müşteri kümeleme işlemi
import joblib, pandas as pd
from sklearn.cluster import KMeans
from data_preprocessing import load_raw, fit_transform
from pathlib import Path    

BASE_DIR  = Path(__file__).resolve().parents[1]   
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Load raw data
RAW = load_raw()
# Transform data
X, _ = fit_transform(RAW)

# Cluster number is 7 so
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
# Cluster etiketi
clusters = kmeans.fit_predict(X)

#
RAW["Cluster_ID"] = clusters
# Save it as excel
RAW.to_excel(DATA_DIR / "hotel_clustered_kmeans.xlsx", index=False)


# kmeans model trained
joblib.dump(kmeans, MODEL_DIR / "kmeans.pkl")     

