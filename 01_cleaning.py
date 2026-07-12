"""
Step 1a: Data Cleaning - In-Vehicle Bar Coupon Subset
Research Question: What contextual and demographic factors drive acceptance of
Bar coupons, and how can this inform a targeted delivery strategy?
"""
import pandas as pd

# ---- Load full dataset ----
df = pd.read_csv("Coupon_Recommendation.csv")
print(f"Full dataset shape: {df.shape}")

# ---- Filter to Bar coupons only (our research scope) ----
# Rationale: Bar coupons have the lowest acceptance rate (41%) of all 5 coupon
# types, making them the most business-relevant subset to investigate.
bar = df[df["coupon"] == "Bar"].copy()
bar = bar.drop(columns=["coupon"])  # constant now, no longer informative
print(f"Bar coupon subset shape: {bar.shape}")

# ---- Check missing values ----
missing = bar.isnull().sum()
print("\nMissing values:\n", missing[missing > 0] if missing.sum() > 0 else "None found")

# ---- Check and handle duplicates ----
# The dataset has no unique respondent ID, so exact duplicate rows could be
# either genuine repeated identical observations (plausible given how coarse/
# categorical most features are) or accidental double-entries. Given the
# duplicate count is very small relative to sample size (7 of 1913 = 0.37%),
# they are dropped to avoid double-counting any single response pattern,
# following standard practice for exact-match duplicate rows.
dupes_before = bar.duplicated().sum()
bar = bar.drop_duplicates()
print(f"\nDropped {dupes_before} duplicate rows. New shape: {bar.shape}")

# ---- Drop redundant / uninformative columns ----
# toCoupon_GEQ5min is constant (=1) for every Bar-coupon row -> no variance,
# no predictive value, safe to drop.
# direction_opp is the exact complement of direction_same -> keep only one to
# avoid multicollinearity.
bar = bar.drop(columns=["toCoupon_GEQ5min", "direction_opp"])
print(f"\nDropped redundant columns. Final shape: {bar.shape}")

# ---- No numeric outlier handling needed ----
# Temperature is the only continuous-style variable, and it only takes 3 fixed
# values (30, 55, 80 F) reflecting discrete survey conditions, not a continuous
# measured variable - so IQR-based outlier detection is not meaningful here.
# All other features are categorical/ordinal by design (survey responses), so
# there are no numeric outliers to treat in this dataset.

# ---- Encode ordinal features for later modelling/EDA ----
freq_map = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
for col in ["Bar", "CoffeeHouse", "CarryAway", "RestaurantLessThan20", "Restaurant20To50"]:
    bar[col + "_ord"] = bar[col].map(freq_map)

age_map = {"below21": 0, "21": 1, "26": 2, "31": 3, "36": 4, "41": 5, "46": 6, "50plus": 7}
bar["age_ord"] = bar["age"].map(age_map)

income_map = {
    "Less than $12500": 0, "$12500 - $24999": 1, "$25000 - $37499": 2,
    "$37500 - $49999": 3, "$50000 - $62499": 4, "$62500 - $74999": 5,
    "$75000 - $87499": 6, "$87500 - $99999": 7, "$100000 or More": 8,
}
bar["income_ord"] = bar["income"].map(income_map)

edu_map = {
    "Some High School": 0, "High School Graduate": 1, "Some college - no degree": 2,
    "Associates degree": 3, "Bachelors degree": 4,
    "Graduate degree (Masters or Doctorate)": 5,
}
bar["education_ord"] = bar["education"].map(edu_map)

print("\nClass balance (target Y):")
print(bar["Y"].value_counts(normalize=True).round(3))

# ---- Save cleaned dataset for EDA / modelling stages ----
bar.to_csv("bar_coupons_clean.csv", index=False)
print("\nSaved cleaned dataset to bar_coupons_clean.csv")
