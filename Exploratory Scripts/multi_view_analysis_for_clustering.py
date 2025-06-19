import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules  # added for market basket analysis
from itertools import combinations
from sklearn.cluster import KMeans

# ————————————————————————————————
# STEP 1: Load raw data
raw_path = r"C:\Users\gunal\Desktop\Bitirme Projesi\final_synced_main_guest_names_dataset.xlsx"
df = pd.read_excel(raw_path)




 # — Label-encode Room Direction and preserve it for pairwise plots
if "Room Direction" in df.columns:
   le = LabelEncoder()
   df["Room Direction"] = le.fit_transform(df["Room Direction"])

# — Add gender mapping right away
if "Guest Gender" in df.columns:
    gender_map = {"Male": 1, "Female": 0}
    df["Guest Gender"] = df["Guest Gender"].map(gender_map).fillna(-1).astype(int)

    

# ————————————————————————————————
# STEP 2: Date features
if "Check-in Date" in df.columns:
    df["Check-in Date"] = pd.to_datetime(df["Check-in Date"], errors="coerce")
    df["Check-in Month"] = df["Check-in Date"].dt.month
    df["Check-in Weekday"] = df["Check-in Date"].dt.weekday
    df.drop("Check-in Date", axis=1, inplace=True)

# ————————————————————————————————
# STEP 3: Categorical → numeric
threshold = 10
df_processed = df.copy()
to_drop = []

for col in df_processed.columns:
    if col in ["Guest Gender", "Room Direction"]:
        continue   # already numeric or specially handled
    if df_processed[col].dtype == "object":
        df_processed[col] = df_processed[col].fillna("Eksik")
        # Try converting to numeric first
        try:
            df_processed[col] = pd.to_numeric(df_processed[col])
            continue
        except:
            pass
        # Low-cardinality → one-hot (keep all dummies so we can refer to them later)
        if df_processed[col].nunique() <= threshold:
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=False)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            to_drop.append(col)
        else:
            # High-cardinality → label encode
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])

# drop original object columns
if to_drop:
    df_processed.drop(columns=to_drop, inplace=True)

# fill numeric NaNs
num_cols = df_processed.select_dtypes(include=[np.number]).columns
df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())

# ————————————————————————————————
# STEP 4: Minibar one-hot
minibar_lists = df["Items Taken from Minibar"].fillna("None").apply(
    lambda x: [] if x.strip().lower()=="none" else [i.strip() for i in x.split(",")]
)
mlb = MultiLabelBinarizer()
encoded = mlb.fit_transform(minibar_lists)
minibar_df = pd.DataFrame(encoded, columns=[f"Minibar_{i}" for i in mlb.classes_])
df_processed = pd.concat([df_processed, minibar_df], axis=1)

# compute a simple Minibar_Count if desired
df_processed["Minibar_Count"] = minibar_df.sum(axis=1)


# ————————————————————————————————
# STEP 4b: Market Basket Analysis on Mini-Bar items
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

print("\n=== Apriori Market Basket Analysis ===")
# use the raw one-hot minibar_df (still 0/1) to find frequent itemsets
frequent_itemsets = apriori(minibar_df, min_support=0.05, use_colnames=True)
rules_ap = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print(frequent_itemsets)
print(rules_ap[['antecedents','consequents','support','confidence','lift']])

print("\n=== FP-Growth Market Basket Analysis ===")
frequent_itemsets_fp = fpgrowth(minibar_df, min_support=0.05, use_colnames=True)
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.7)
print(frequent_itemsets_fp)
print(rules_fp[['antecedents','consequents','support','confidence','lift']])

N = len(minibar_df)

# 1) Popularity: top 10 individual items by support
item_support = minibar_df.sum(axis=0) / N
top10_items = item_support.sort_values(ascending=False).head(12)

plt.figure(figsize=(8,5))
plt.barh(range(len(top10_items)), top10_items.values, align='center')
plt.yticks(range(len(top10_items)), top10_items.index)
plt.gca().invert_yaxis()
plt.xlabel('Support (Popularity)')
plt.title('Top 10 Minibar Items by Popularity')
plt.tight_layout()
plt.show()

# 2) Correlation heatmap of item co-occurrence
corr = minibar_df.corr()

plt.figure(figsize=(8,8))
plt.matshow(corr, fignum=1, cmap='coolwarm')
plt.colorbar()
labels = minibar_df.columns
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
plt.title('Correlation Matrix of Minibar Items', pad=20)
plt.tight_layout()
plt.show()

# 3) Top 10 item-pairs by co-occurrence support
pair_supports = []
for a, b in combinations(minibar_df.columns, 2):
    support = ((minibar_df[a] & minibar_df[b]).sum()) / N
    pair_supports.append(((a, b), support))
pair_supports.sort(key=lambda x: x[1], reverse=True)
top10_pairs = pair_supports[:10]

labels_pairs = [f"{a} & {b}" for (a, b), _ in top10_pairs]
values_pairs = [sup for _, sup in top10_pairs]

plt.figure(figsize=(8,5))
plt.barh(range(len(values_pairs)), values_pairs, align='center')
plt.yticks(range(len(values_pairs)), labels_pairs)
plt.gca().invert_yaxis()
plt.xlabel('Support (Co-occurrence)')
plt.title('Top 10 Minibar Item Pairs by Co-occurrence Support')
plt.tight_layout()
plt.show()
# ————————————————————————————————
# STEP 5: Scale everything
scaler = StandardScaler()
num_cols = df_processed.select_dtypes(include=[np.number]).columns
df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

# — Optional: sanity-check your columns
print("Preprocessed columns:", df_processed.columns.tolist())



    # ————————————————————————————————
    # STEP 5b: Hot/Cold & Light/Dark & Family Clustering
    # 1) Hot/Cold preference (2 clusters)

X_heat = df_processed[['Preferred Heat (°C)']].values
km_heat = KMeans(n_clusters=2, random_state=42).fit(X_heat)
df_processed['Cluster_Heat'] = km_heat.labels_
    # visualize
plt.figure()
plt.hist([X_heat[km_heat.labels_==0].flatten(),
              X_heat[km_heat.labels_==1].flatten()],
             label=['Cold Lovers','Hot Lovers'], bins=20)
plt.legend(); plt.xlabel('Preferred Heat (°C)'); plt.title('Hot vs Cold Clusters')
plt.show()

    # 2) Light/Dark preference (2 clusters)
X_light = df_processed[['Preferred Dimmer Level','Main Room Illumination (Lumens)']].values
km_light = KMeans(n_clusters=2, random_state=42).fit(X_light)
df_processed['Cluster_Light'] = km_light.labels_
    # visualize
plt.figure()
plt.scatter(X_light[:,0], X_light[:,1], c=km_light.labels_, cmap='tab10', s=30)
plt.xlabel('Preferred Dimmer Level'); plt.ylabel('Main Room Illumination (Lumens)')
plt.title('Light vs Dark Clusters'); plt.show()

    # 3) Family flag + their own clusters
df_processed['IsFamily'] = df_processed['Room Number'].astype(str).str.startswith('N')
    # a) Family Hot/Cold
fam = df_processed


   # ————————————————————————————————
   # STEP 5c: Gender vs. Preferred Heat clustering & plot
X_gh = df_processed[['Guest Gender', 'Preferred Heat (°C)']].dropna().values
km_gh = KMeans(n_clusters=2, random_state=42).fit(X_gh)
df_processed['Cluster_GenderHeat'] = km_gh.labels_
# Plot with a bit of jitter on gender
jitter = (np.random.rand(len(X_gh)) - .5) * .1
plt.figure(figsize=(6,5))
plt.scatter(
df_processed['Guest Gender'] + jitter,
df_processed['Preferred Heat (°C)'],
c=km_gh.labels_, cmap='tab10', s=30, alpha=0.7
)
centers = km_gh.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', marker='X', s=100, label='Centers')
plt.xlabel('Guest Gender (0=Female,1=Male)')
plt.ylabel('Preferred Heat (°C)')
plt.title('Clusters: Gender vs. Preferred Heat')
plt.legend()
plt.tight_layout()
plt.show()


# ————————————————————————————————
# STEP 6: Define your exact pairs (must match df_processed columns now!)
pairs = {
    "gender_heat":        ["Guest Gender",                 "Preferred Heat (°C)"],
    "direction_temp":     ["Room Direction",               "Avg Temp Without Cooling (°C)"],
    "direction_cooling":  ["Room Direction",               "Cooling Cost (Reformed)"],
    "dimmer_mainillum":   ["Preferred Dimmer Level",       "Main Room Illumination (Lumens)"],
    "temp_cooling":       ["Avg Temp Without Cooling (°C)","Cooling Cost (Reformed)"],
    "bath_main_illum":    ["Bathroom Illumination (Lumens)","Main Room Illumination (Lumens)"],
    "cooling_illum":      ["Cooling Cost (Reformed)",      "Main Room Illumination (Lumens)"],
    "guestcount_minibar": ["Guests Count",                 "Minibar_Count"]
}

# ————————————————————————————————
# STEP 7: Pairwise clustering + 1×3 subplot for each
for name, (f1, f2) in pairs.items():
    # verify existence
    if not {f1, f2}.issubset(df_processed.columns):
        print(f"→ Skipping {name}: {f1!r} or {f2!r} not in preprocessed columns")
        continue

    X = df_processed[[f1, f2]].values
    # already scaled, so reuse or rescale small slice:
    X = scaler.fit_transform(X)

    algos = {
        "KMeans":       KMeans(n_clusters=4, random_state=123),
        "DBSCAN":       DBSCAN(eps=0.5, min_samples=10),
        "Agglomerative": AgglomerativeClustering(n_clusters=4)
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (algo_name, algo) in zip(axes, algos.items()):
        labels = algo.fit_predict(X)
        colname = f"cluster_{algo_name.lower()}_{name}"
        df_processed[colname] = labels

        sc = ax.scatter(X[:,0], X[:,1], c=labels, cmap="tab10", s=30)
        ax.set_title(f"{algo_name}: {f1} vs {f2}")
        ax.set_xlabel(f1); ax.set_ylabel(f2)
        fig.colorbar(sc, ax=ax, label="Cluster")

    plt.suptitle(f"Pairwise clusters for '{name}'")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

# ————————————————————————————————
# STEP 8: Export
out = r"C:\Users\gunal\Desktop\Bitirme Projesi\hotel_clustered_pairs.xlsx"
df_processed.to_excel(out, index=False)
print("Saved all to", out)