"""
Step 1b: Exploratory Data Analysis - Bar Coupon Acceptance
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
bar = pd.read_csv("bar_coupons_clean.csv")

# ---- Overall acceptance rate ----
overall_rate = bar["Y"].mean()
print(f"Overall Bar coupon acceptance rate: {overall_rate:.1%}")

# ---- Statistical summary of ordinal/numeric features ----
ord_cols = ["Bar_ord", "CoffeeHouse_ord", "CarryAway_ord", "RestaurantLessThan20_ord",
            "Restaurant20To50_ord", "age_ord", "income_ord", "education_ord", "temperature"]
print("\nStatistical summary:\n", bar[ord_cols].describe().round(2))

# =========================================================
# Chart 1: Acceptance rate by own Bar-visit frequency
# (hypothesis: prior bar-going habit is the strongest driver)
# =========================================================
freq_order = ["never", "less1", "1~3", "4~8", "gt8"]
rate_by_barfreq = bar.groupby("Bar")["Y"].mean().reindex(freq_order)

plt.figure(figsize=(8, 5))
rate_by_barfreq.plot(kind="bar", color="#4C72B0", edgecolor="black")
plt.axhline(overall_rate, color="red", linestyle="--", label=f"Overall rate ({overall_rate:.0%})")
plt.title("Bar Coupon Acceptance Rate by Prior Bar-Visit Frequency")
plt.ylabel("Acceptance Rate")
plt.xlabel("Self-reported Bar visits (frequency)")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig("chart1_acceptance_by_barfreq.png", dpi=150)
plt.close()

# =========================================================
# Chart 2: Acceptance rate by passenger type and expiration window
# (hypothesis: social context + urgency matter)
# =========================================================
pivot = bar.pivot_table(index="passanger", columns="expiration", values="Y", aggfunc="mean")
plt.figure(figsize=(7, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Acceptance Rate"})
plt.title("Acceptance Rate by Passenger Type and Coupon Expiration")
plt.tight_layout()
plt.savefig("chart2_passenger_expiration_heatmap.png", dpi=150)
plt.close()

# =========================================================
# Chart 3: Acceptance rate by time of day
# =========================================================
time_order = ["7AM", "10AM", "2PM", "6PM", "10PM"]
rate_by_time = bar.groupby("time")["Y"].mean().reindex(time_order)
plt.figure(figsize=(8, 5))
rate_by_time.plot(kind="bar", color="#55A868", edgecolor="black")
plt.axhline(overall_rate, color="red", linestyle="--", label=f"Overall rate ({overall_rate:.0%})")
plt.title("Bar Coupon Acceptance Rate by Time of Day")
plt.ylabel("Acceptance Rate")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig("chart3_acceptance_by_time.png", dpi=150)
plt.close()

# =========================================================
# Chart 4: Acceptance rate by age group and marital/children status
# =========================================================
plt.figure(figsize=(9, 5))
sns.barplot(data=bar, x="age", y="Y", hue="has_children",
            order=["below21", "21", "26", "31", "36", "41", "46", "50plus"],
            estimator="mean", errorbar=None, palette="muted")
plt.axhline(overall_rate, color="red", linestyle="--", label=f"Overall rate ({overall_rate:.0%})")
plt.title("Acceptance Rate by Age Group and Presence of Children")
plt.ylabel("Acceptance Rate")
plt.xlabel("Age Group")
plt.legend(title="Has Children")
plt.tight_layout()
plt.savefig("chart4_age_children.png", dpi=150)
plt.close()

# =========================================================
# Chart 5: Correlation heatmap of ordinal-encoded features vs Y
# =========================================================
corr = bar[ord_cols + ["Y"]].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Between Ordinal Features and Coupon Acceptance")
plt.tight_layout()
plt.savefig("chart5_correlation_heatmap.png", dpi=150)
plt.close()

# ---- Print key numeric findings for the report narrative ----
print("\n--- Key findings for write-up ---")
print("Acceptance by Bar-visit frequency:\n", rate_by_barfreq.round(3))
print("\nAcceptance by time of day:\n", rate_by_time.round(3))
print("\nCorrelation of each feature with Y:\n", corr["Y"].drop("Y").sort_values(ascending=False).round(3))

print("\nAll charts saved.")
