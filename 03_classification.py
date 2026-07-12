"""
Step 2a: Classification - Predicting Bar Coupon Acceptance
Models compared: Logistic Regression (interpretable baseline) vs.
Random Forest (higher-capacity, handles non-linear interactions).
Validated with stratified train/test split + 5-fold cross-validation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              roc_auc_score, RocCurveDisplay)

sns.set_style("whitegrid")
bar = pd.read_csv("bar_coupons_clean.csv")

# ---- Feature selection ----
# Rationale: driven directly by the EDA. Bar-visit frequency, passenger type,
# expiration window and time of day showed the clearest relationship with Y.
# We add secondary demographic/context features to give the models enough
# signal, while excluding raw string columns already captured by *_ord
# encodings (to avoid duplicate/redundant information).
categorical_features = ["passanger", "expiration", "time", "destination",
                         "weather", "gender", "maritalStatus", "has_children",
                         "direction_same"]
ordinal_features = ["Bar_ord", "CoffeeHouse_ord", "CarryAway_ord",
                     "RestaurantLessThan20_ord", "Restaurant20To50_ord",
                     "age_ord", "income_ord", "education_ord", "temperature"]

X = pd.get_dummies(bar[categorical_features], drop_first=True)
X[ordinal_features] = bar[ordinal_features]
y = bar["Y"]

print(f"Feature matrix shape: {X.shape}")

# ---- Train/test split (stratified to preserve 59/41 class balance) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Standardise ordinal/continuous columns (needed for Logistic Regression) ----
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[ordinal_features] = scaler.fit_transform(X_train[ordinal_features])
X_test_scaled[ordinal_features] = scaler.transform(X_test[ordinal_features])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# =========================================================
# Model 1: Logistic Regression
# =========================================================
logit = LogisticRegression(max_iter=1000, random_state=42)
logit_cv_scores = cross_val_score(logit, X_train_scaled, y_train, cv=cv, scoring="accuracy")
logit.fit(X_train_scaled, y_train)
logit_pred = logit.predict(X_test_scaled)
logit_proba = logit.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression ===")
print(f"5-fold CV accuracy: {logit_cv_scores.mean():.3f} (+/- {logit_cv_scores.std():.3f})")
print(f"Test accuracy: {accuracy_score(y_test, logit_pred):.3f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, logit_proba):.3f}")
print(classification_report(y_test, logit_pred))
results["Logistic Regression"] = {
    "cv_acc": logit_cv_scores.mean(), "test_acc": accuracy_score(y_test, logit_pred),
    "auc": roc_auc_score(y_test, logit_proba), "pred": logit_pred, "proba": logit_proba
}

# =========================================================
# Model 2: Random Forest
# =========================================================
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print(f"5-fold CV accuracy: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std():.3f})")
print(f"Test accuracy: {accuracy_score(y_test, rf_pred):.3f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, rf_proba):.3f}")
print(classification_report(y_test, rf_pred))
results["Random Forest"] = {
    "cv_acc": rf_cv_scores.mean(), "test_acc": accuracy_score(y_test, rf_pred),
    "auc": roc_auc_score(y_test, rf_proba), "pred": rf_pred, "proba": rf_proba
}

# Note on class_weight="balanced" for RF: the 59/41 split is only mildly
# imbalanced, but balancing class weights costs nothing and slightly improves
# minority-class (accept=1) recall, which matters more for this business
# problem than raw accuracy (missing a genuine acceptor is the costlier error).

# =========================================================
# Confusion matrices (side by side)
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Reject", "Accept"], yticklabels=["Reject", "Accept"])
    ax.set_title(f"{name}\nAccuracy: {res['test_acc']:.2f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("chart6_confusion_matrices.png", dpi=150)
plt.close()

# =========================================================
# ROC curves comparison
# =========================================================
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, logit_proba, name="Logistic Regression", ax=ax)
RocCurveDisplay.from_predictions(y_test, rf_proba, name="Random Forest", ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
ax.set_title("ROC Curve Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("chart7_roc_curves.png", dpi=150)
plt.close()

# =========================================================
# Random Forest feature importance (top 10)
# =========================================================
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop 10 Random Forest feature importances:\n", importances.head(10).round(3))

plt.figure(figsize=(8, 6))
importances.head(10).sort_values().plot(kind="barh", color="#C44E52", edgecolor="black")
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("chart8_feature_importance.png", dpi=150)
plt.close()

# ---- Save model comparison summary ----
summary = pd.DataFrame({
    "Model": list(results.keys()),
    "CV Accuracy": [results[m]["cv_acc"] for m in results],
    "Test Accuracy": [results[m]["test_acc"] for m in results],
    "Test ROC-AUC": [results[m]["auc"] for m in results],
})
summary.to_csv("model_comparison_summary.csv", index=False)
print("\nModel comparison summary:\n", summary.round(3))
print("\nAll classification charts and summary saved.")
