"""
Step 3: Translating models into a KPI-tied targeting strategy.
Produces:
1. A decile gains/lift table from the Random Forest's predicted acceptance
   probability (out-of-fold, unbiased) - the standard marketing tool for
   deciding what % of the audience to target.
2. A cluster-based send/suppress simulation quantifying the volume and
   redemption-rate trade-off of excluding the lowest-propensity segment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

sns.set_style("whitegrid")
bar = pd.read_csv("bar_coupons_with_clusters.csv")

categorical_features = ["passanger", "expiration", "time", "destination",
                         "weather", "gender", "maritalStatus", "has_children",
                         "direction_same"]
ordinal_features = ["Bar_ord", "CoffeeHouse_ord", "CarryAway_ord",
                     "RestaurantLessThan20_ord", "Restaurant20To50_ord",
                     "age_ord", "income_ord", "education_ord", "temperature"]

X = pd.get_dummies(bar[categorical_features], drop_first=True)
X[ordinal_features] = bar[ordinal_features]
y = bar["Y"]

# ---- Out-of-fold predicted probabilities (unbiased estimate for the whole
# dataset, avoids the optimism of scoring the model on data it was trained on) ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
oof_proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
bar["pred_proba"] = oof_proba

# =========================================================
# Decile gains table
# =========================================================
bar["decile"] = pd.qcut(bar["pred_proba"], 10, labels=False, duplicates="drop")
bar["decile"] = 9 - bar["decile"]  # 0 = highest propensity decile

gains = bar.groupby("decile").agg(
    n=("Y", "size"),
    acceptance_rate=("Y", "mean"),
    total_acceptors=("Y", "sum")
).sort_index()
gains["cum_acceptors"] = gains["total_acceptors"].cumsum()
gains["cum_pct_of_all_acceptors"] = (gains["cum_acceptors"] / gains["total_acceptors"].sum() * 100).round(1)
gains["cum_pct_of_audience"] = ((gains["n"].cumsum()) / gains["n"].sum() * 100).round(1)
gains["lift_vs_baseline"] = (gains["acceptance_rate"] / bar["Y"].mean()).round(2)

print("Decile gains table (decile 0 = highest predicted propensity):\n")
print(gains.round(3))
gains.to_csv("gains_table.csv")

# Key headline stat: what % of all acceptors are captured in top 4 deciles (40% of audience)?
top4_capture = gains.loc[0:3, "cum_pct_of_all_acceptors"].iloc[-1]
top4_audience = gains.loc[0:3, "cum_pct_of_audience"].iloc[-1]
print(f"\nHeadline: targeting the top {top4_audience}% of drivers by predicted "
      f"propensity captures {top4_capture}% of all coupon acceptances.")

# =========================================================
# Gains chart (cumulative acceptors captured vs. % audience targeted)
# =========================================================
plt.figure(figsize=(7.5, 6))
x_vals = [0] + gains["cum_pct_of_audience"].tolist()
y_vals = [0] + gains["cum_pct_of_all_acceptors"].tolist()
plt.plot(x_vals, y_vals, marker="o", color="#4C72B0", label="Model-targeted sends")
plt.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random targeting (no model)")
plt.xlabel("% of Drivers Targeted (ranked by predicted propensity)")
plt.ylabel("% of All Coupon Acceptances Captured")
plt.title("Gains Chart: Model-Based Targeting vs. Random Sending")
plt.legend()
plt.tight_layout()
plt.savefig("chart12_gains_chart.png", dpi=150)
plt.close()

# =========================================================
# Cluster-based send/suppress simulation
# =========================================================
overall_rate = bar["Y"].mean()
cluster_stats = bar.groupby("Cluster").agg(n=("Y", "size"), acceptance_rate=("Y", "mean")).sort_index()
cluster_stats["share_of_volume_pct"] = (cluster_stats["n"] / cluster_stats["n"].sum() * 100).round(1)

# Simulate: suppress sends to the lowest-acceptance cluster (Cluster 1)
suppress_cluster = cluster_stats["acceptance_rate"].idxmin()
kept = bar[bar["Cluster"] != suppress_cluster]
sends_saved_pct = (bar["Cluster"] == suppress_cluster).mean() * 100
new_rate = kept["Y"].mean()
acceptors_lost_pct = (1 - kept["Y"].sum() / bar["Y"].sum()) * 100

print(f"\n--- Send/Suppress Simulation ---")
print(f"Suppressing Cluster {suppress_cluster} (lowest acceptance rate "
      f"{cluster_stats.loc[suppress_cluster, 'acceptance_rate']:.1%}):")
print(f"  Reduces total coupon sends by {sends_saved_pct:.1f}%")
print(f"  Raises acceptance rate among remaining sends from {overall_rate:.1%} to {new_rate:.1%}")
print(f"  Forgoes only {acceptors_lost_pct:.1f}% of total acceptances")

cluster_stats.to_csv("cluster_send_suppress_stats.csv")

print("\nAll recommendation-support charts and tables saved.")
