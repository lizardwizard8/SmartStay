# Clustering methods using scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from pycaret.clustering import setup, create_model, evaluate_model, plot_model, assign_model

# 1. Load raw dataset and prepare features
raw_file = "C:/Users/gunal/Desktop/Bitirme Projesi/hotel_dataset_with_reformed_cooling.xlsx"
df = pd.read_excel(raw_file)

# 1.1 Process date columns if present
if "Check-in Date" in df.columns:
    # convert text to datetime, extract month and weekday, then drop original
    df["Check-in Date"] = pd.to_datetime(df["Check-in Date"], errors="coerce")
    df["Check-in Month"] = df["Check-in Date"].dt.month
    df["Check-in Weekday"] = df["Check-in Date"].dt.weekday
    df.drop(["Check-in Date"], axis=1, inplace=True)

# 1.2 Convert categorical data to numeric
threshold = 10
df_processed = df.copy()
drop_cols = []
for col in df_processed.columns:
    if df_processed[col].dtype == 'object':
        # fill missing text entries with a placeholder
        df_processed[col] = df_processed[col].fillna("Missing")
        try:
            # try converting text directly to numeric values
            df_processed[col] = pd.to_numeric(df_processed[col])
            continue
        except:
            pass
        if df_processed[col].nunique() <= threshold:
            # create dummy variables for low-cardinality categories
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            drop_cols.append(col)
        else:
            # use label encoding for high-cardinality categories
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
# remove original text columns after encoding
if drop_cols:
    df_processed.drop(columns=drop_cols, inplace=True)

# 1.3 Fill missing numeric values with the column median
num_cols = df_processed.select_dtypes(include=[np.number]).columns
df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())

# 1.4 Encode minibar items into separate binary columns
minibar_lists = df["Items Taken from Minibar"].fillna("None").apply(
    lambda x: [] if x.strip().lower() == "none" else [item.strip() for item in x.split(',')]
)
mlb = MultiLabelBinarizer()
minibar_encoded = mlb.fit_transform(minibar_lists)
minibar_df = pd.DataFrame(minibar_encoded, columns=[f"Minibar_{item}" for item in mlb.classes_])
# add minibar features to processed dataframe
df_processed = pd.concat([df_processed, minibar_df], axis=1)

# 1.5 Standardize numeric features to have zero mean and unit variance
num_cols = df_processed.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

# 2. Remove columns that are not relevant for clustering
uncorr_features = [
    'Booking ID',
    'Room Number',
    'Check-in Month',
    'Check-in Weekday'
]
df_clustering = df_processed.drop(columns=[c for c in uncorr_features if c in df_processed.columns])

# 3. Perform K-Means clustering with PyCaret
exp1 = setup(data=df_clustering, normalize=False, session_id=123)
kmeans_model = create_model('kmeans', num_clusters=7)
plot_model(kmeans_model, plot='cluster')
df_kmeans = assign_model(kmeans_model)
# save clustering results to Excel
df_kmeans.to_excel("hotel_clustered_kmeans.xlsx", index=False)

# 4. Perform DBSCAN clustering and visualize distance plot
k = 10
neighbors = NearestNeighbors(n_neighbors=k).fit(df_clustering)
distances, _ = neighbors.kneighbors(df_clustering)
k_distances = np.sort(distances[:, k-1])
plt.figure(figsize=(8,4))
plt.plot(k_distances)
plt.xlabel('Ordered data points')
plt.ylabel(f'{k}th nearest neighbor distance')
plt.title('DBSCAN distance plot')
plt.show()
dbscan_model = create_model('dbscan', eps=3.8, min_samples=10)
evaluate_model(dbscan_model)
plot_model(dbscan_model, plot='cluster')
df_dbscan = assign_model(dbscan_model)
# save DBSCAN results to Excel
df_dbscan.to_excel("hotel_clustered_dbscan.xlsx", index=False)

# 5. Perform hierarchical clustering and save results
exp2 = setup(data=df_clustering, normalize=False, session_id=123)
agglom_model = create_model('hclust', num_clusters=4, method='average', affinity='euclidean')
evaluate_model(agglom_model)
plot_model(agglom_model, plot='cluster')
df_agglom = assign_model(agglom_model)
df_agglom.to_excel("hotel_clustered_agglom.xlsx", index=False)
