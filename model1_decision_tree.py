import pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

df = pd.read_csv("data/superstore.csv", encoding="latin-1")
df['Target'] = (df['Profit'] > 0).astype(int)

features_num = ['Sales','Quantity','Discount','Shipping Cost']
features_cat = ['Segment','Market','Region','Category','Sub-Category','Order Priority','Ship Mode']

X = df[features_num + features_cat].copy()
y = df['Target']

for c in features_cat:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:,1]

results = {
    "n_train": len(X_train), "n_test": len(X_test),
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}
fi = sorted(zip(X.columns, clf.feature_importances_), key=lambda x: -x[1])
results["feature_importance"] = [(f, round(v,4)) for f,v in fi]
print(json.dumps(results, indent=2))

with open("results_model1.json","w") as f:
    json.dump(results, f, indent=2)
