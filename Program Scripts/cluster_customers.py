# Müşteri kümeleme işlemi
import joblib, pandas as pd
from sklearn.cluster import KMeans
from data_preprocessing import load_raw, fit_transform

# Ham veriyi yükle
RAW = load_raw()
# Veriyi dönüştür ve özellik matrisini oluştur
X, _ = fit_transform(RAW)

# Küme sayısını 7 olarak belirleyerek KMeans modelini oluştur
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
# Veriye küme etiketlerini uygula
clusters = kmeans.fit_predict(X)

# Oluşan küme etiketlerini orijinal veriye ekle
RAW["Cluster_ID"] = clusters
# Sonucu Excel dosyası olarak kaydet
RAW.to_excel("hotel_clustered_kmeans.xlsx", index=False)

# Eğitilmiş KMeans modelini dosyaya yaz
joblib.dump(kmeans, "models/kmeans.pkl")
