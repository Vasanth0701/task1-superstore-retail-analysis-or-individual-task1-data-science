# Individual Task 1 Part 1 — Analysis Code

- `model1_decision_tree.py` — Decision Tree classifier on the Global Superstore
  Sales dataset (51,290 rows), predicting Profitable vs Loss-making orders.
- `model2_knn.py` — Builds RFM-style customer features from the UCI Online
  Retail transaction dataset (541,909 rows, 4,338 customers) and trains a
  k-Nearest Neighbours classifier to predict High-Value customers.

Run with: `pip install pandas numpy scikit-learn` then `python3 model1_decision_tree.py`
and `python3 model2_knn.py`. Both scripts download-free once the two CSVs below
are placed in a `data/` folder next to the scripts:

- Superstore: https://raw.githubusercontent.com/pplonski/datasets-for-start/master/superstore-sales/superstore_dataset2011-2015.csv
- Online Retail: https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv

Results (JSON) are written to `results_model1.json` and `results_model2.json`.
Upload this folder to your own GitHub repo and link it in Appendix B of the report.
