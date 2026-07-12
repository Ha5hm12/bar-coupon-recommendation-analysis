# Bar Coupon Recommendation Analysis

Individual assessment analysing the In-Vehicle Coupon Recommendation dataset 
to investigate why Bar coupons have the lowest acceptance rate (41.2%) of all 
five coupon types, and to build a targeted delivery strategy using 
classification and segmentation.

## Research Question
What contextual and demographic factors drive acceptance of Bar coupons among 
drivers, and how can this be used to build a targeted delivery strategy that 
improves the Bar coupon redemption rate above its current 41% baseline?

## How to run
Scripts are numbered and should be run in order:

1. `01_cleaning.py` — filters to Bar coupons, handles duplicates, encodes features
2. `02_eda.py` — exploratory analysis and charts 1–5
3. `03_classification.py` — Logistic Regression vs Random Forest, charts 6–8
4. `04_clustering.py` — K-Means segmentation with elbow/silhouette validation, charts 9–11
5. `05_recommendations.py` — gains chart and targeting simulation, chart 12

Each script reads the CSV output of the previous step and writes its own 
outputs (cleaned CSVs, result tables, PNG charts) into the same folder.

## Contents
- `*.py` — analysis scripts (see above)
- `bar_coupons_clean.csv` / `bar_coupons_with_clusters.csv` — cleaned data
- `model_comparison_summary.csv`, `cluster_profile.csv`, `gains_table.csv` — result tables
- `chart1–12*.png` — all figures referenced in the report

## Key findings
- Random Forest classifier: 76.7% test accuracy, 0.849 ROC-AUC
- Strongest predictor: prior Bar-visit frequency (feature importance 0.25)
- K-Means segmentation (K=4) identified a high-propensity cluster (57.5% 
  acceptance) and a low-propensity cluster (27.1% acceptance)
- Propensity-based targeting on the top 40% of drivers captures 69.8% of all 
  coupon acceptances
