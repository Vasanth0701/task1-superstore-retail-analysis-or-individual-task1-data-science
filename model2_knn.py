import pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv("data/online_retail_full.csv", encoding="latin-1")
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity']>0) & (df['UnitPrice']>0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Revenue'] = df['Quantity'] * df['UnitPrice']

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('Revenue', 'sum'),
    AvgBasket=('Revenue', 'mean'),
    DistinctItems=('StockCode', 'nunique')
).reset_index()

print("Customers:", rfm.shape)
print(rfm.describe())

# Target: High-value customer = top 25% by Monetary spend
thresh = rfm['Monetary'].quantile(0.75)
rfm['HighValue'] = (rfm['Monetary'] >= thresh).astype(int)
print("Positive class rate:", rfm['HighValue'].mean())

# Use Recency & Frequency & DistinctItems & AvgBasket (NOT Monetary, to avoid leakage/tautology) to predict HighValue
X = rfm[['Recency','Frequency','AvgBasket','DistinctItems']]
y = rfm['HighValue']

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)

clf = KNeighborsClassifier(n_neighbors=9, weights='distance')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:,1]

results = {
    "n_customers": int(rfm.shape[0]),
    "n_train": len(X_train), "n_test": len(X_test),
    "positive_rate": float(y.mean()),
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}
print(json.dumps(results, indent=2))
with open("results_model2.json","w") as f:
    json.dump(results, f, indent=2)
rfm.to_csv("rfm_customers.csv", index=False)
