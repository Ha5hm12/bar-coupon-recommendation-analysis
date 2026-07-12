"""
Step 2b: Segmentation - K-Means Clustering on Bar Coupon Respondents
Goal: identify distinct behavioural/contextual segments to translate into a
targeted send/suppress coupon strategy.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
bar = pd.read_csv("bar_coupons_clean.csv")

# ---- Feature selection for clustering ----
# Rationale: we cluster on behavioural/lifestyle + demographic features (not
# on Y itself, and not on situational one-off context like weather/time,
# which describe the trip rather than the person). This produces segments
# that are stable and usable for a targeting policy (i.e. "who to send Bar
# coupons to"), rather than clusters that just reproduce single trips.
cluster_features = ["Bar_ord", "CoffeeHouse_ord", "CarryAway_ord",
                     "RestaurantLessThan20_ord", "Restaurant20To50_ord",
                     "age_ord", "income_ord", "education_ord"]

X = bar[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# Justify K: Elbow method (inertia) + Silhouette score across K=2..8
# =========================================================
inertias = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(K_range), inertias, marker="o", color="#4C72B0")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(K_range), sil_scores, marker="o", color="#55A868")
axes[1].set_title("Silhouette Score by K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("chart9_k_selection.png", dpi=150)
plt.close()

print("Inertia by K:", dict(zip(K_range, np.round(inertias, 1))))
print("Silhouette by K:", dict(zip(K_range, np.round(sil_scores, 3))))

best_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"\nK with highest silhouette score: {best_k}")

# ---- Final clustering ----
# K=4 chosen: silhouette score is at/near its peak, the elbow plot visibly
# flattens from K=4 onward, and 4 segments is a practically usable number for
# a targeting policy (too many clusters would fragment the marketing rules
# into something impractical to act on).
K_FINAL = 4
kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
bar["Cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nFinal silhouette score at K={K_FINAL}: {silhouette_score(X_scaled, bar['Cluster']):.3f}")

# =========================================================
# Cluster profiling: mean feature values + acceptance rate per cluster
# =========================================================
profile = bar.groupby("Cluster")[cluster_features + ["Y"]].mean().round(2)
profile["n"] = bar["Cluster"].value_counts().sort_index()
print("\nCluster profile (mean feature values + acceptance rate):\n", profile)
profile.to_csv("cluster_profile.csv")

# =========================================================
# Visualise clusters via PCA (2D projection) coloured by cluster and by Y
# =========================================================
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_scaled)
bar["PC1"], bar["PC2"] = pcs[:, 0], pcs[:, 1]
print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_.round(3)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
sns.scatterplot(data=bar, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=axes[0], alpha=0.7)
axes[0].set_title("K-Means Clusters (PCA projection)")

sns.scatterplot(data=bar, x="PC1", y="PC2", hue="Y", palette=["#C44E52", "#55A868"], ax=axes[1], alpha=0.6)
axes[1].set_title("Actual Coupon Acceptance (PCA projection)")
plt.tight_layout()
plt.savefig("chart10_cluster_pca.png", dpi=150)
plt.close()

# =========================================================
# Acceptance rate by cluster (bar chart) - the key business chart
# =========================================================
overall_rate = bar["Y"].mean()
acc_by_cluster = bar.groupby("Cluster")["Y"].mean().sort_values(ascending=False)

plt.figure(figsize=(7, 5))
acc_by_cluster.plot(kind="bar", color="#8172B2", edgecolor="black")
plt.axhline(overall_rate, color="red", linestyle="--", label=f"Overall rate ({overall_rate:.0%})")
plt.title("Bar Coupon Acceptance Rate by Segment")
plt.ylabel("Acceptance Rate")
plt.xlabel("Cluster")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig("chart11_acceptance_by_cluster.png", dpi=150)
plt.close()

bar.to_csv("bar_coupons_with_clusters.csv", index=False)
print("\nAll clustering charts and cluster-labelled dataset saved.")
