# minibar_rules.py – map each cluster to its top minibar items
import pandas as pd
import joblib, sys
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path   


BASE_DIR  = Path(__file__).resolve().parents[1]   # <Bitirme-Projesi>
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)                   

# Configuration parameters
DATA_PATH    = DATA_DIR  / "final_synced_main_guest_names_dataset.xlsx"   # 
TXT_COL      = "Items Taken from Minibar"
CLUSTER_FILE = DATA_DIR  / "hotel_clustered_kmeans.xlsx"
OUTFILE      = MODEL_DIR / "minibar_rules.pkl" 
TOP_N        = 3       # number of items to select per cluster
MIN_SUPPORT  = 0.01    # minimum support for frequent itemsets
LIFT_THRESH  = 1.0     # minimum lift for association rules

# 1) Read raw dataset and confirm minibar column exists
df_raw = pd.read_excel(DATA_PATH)
if TXT_COL not in df_raw.columns:
    sys.exit(f"[!] {TXT_COL!r} not found in {DATA_PATH}")

# Function to split and clean minibar entries into a list
def parse_items(cell):
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text or text.lower() == "none":
        return []
    parts = {seg.strip() for seg in text.split(",") if seg.strip()}
    return sorted(parts)

# Convert each row into a list of minibar items
transactions = df_raw[TXT_COL].apply(parse_items).tolist()

# 2) One-hot encode the minibar transactions
encoder = TransactionEncoder()
ohe_array = encoder.fit(transactions).transform(transactions)
ohe = pd.DataFrame(
    ohe_array,
    columns=encoder.columns_,
    index=df_raw.index
).astype(int)

# 3) Load cluster labels and add a flag column for each cluster
clusters = pd.read_excel(CLUSTER_FILE)["Cluster_ID"].reset_index(drop=True)
if len(clusters) != len(ohe):
    sys.exit("[!] Row count mismatch between clusters and transactions")
for cid in clusters.unique():
    ohe[f"Cluster_{cid}"] = (clusters == cid).astype(int)

# 4) Find association rules from the encoded data
frequent = apriori(ohe, min_support=MIN_SUPPORT, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=LIFT_THRESH)
if rules.empty:
    # if no rules meet the lift threshold, use confidence instead
    rules = association_rules(frequent, metric="confidence", min_threshold=0.2)

# 5) Build cluster-to-items mapping with top items by lift
cluster_to_items = {}
for cid in clusters.unique():
    flag = f"Cluster_{cid}"
    subset = rules[rules["antecedents"].apply(lambda ants: flag in ants)]
    subset = subset.sort_values("lift", ascending=False)
    items = []
    for conseq in subset["consequents"]:
        if len(conseq) == 1:
            item = next(iter(conseq)).replace("Minibar_", "")
            if item not in items:
                items.append(item)
        if len(items) >= TOP_N:
            break
    # default to sparkling water if no items found
    cluster_to_items[int(cid)] = items or ["Sparkling Water"]

# 6) Save the rules mapping for later use
joblib.dump(cluster_to_items, OUTFILE)
print(f"[✓] Saved minibar rules for {len(cluster_to_items)} clusters to {OUTFILE}")
