"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Author : Palarapu Harshavardhan Reddy (Information Technology Department)

Reference dataset: DataCo Smart Supply Chain (Kaggle). Column names below follow
a simplified version of that dataset. Because the full file is not bundled, a
synthetic sample with deliberately injected defects is generated so the whole
pipeline can be run end to end. To use the real file, replace generate_sample()
with load_data("DataCoSupplyChainDataset.csv") and map the column names.

Requires: pandas, numpy, scikit-learn (matplotlib optional, for charts)
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_data(path):
    """Step 1 - Data collection: read the raw CSV (the DataCo file is latin-1 encoded)."""
    return pd.read_csv(path, encoding="latin-1")


def generate_sample(n=5000, seed=42):
    """Simulate a raw logistics extract with realistic quality problems."""
    rng = np.random.default_rng(seed)
    modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    scheduled_map = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}
    regions = ["Western Europe", "Central America", "South America", "Oceania",
               "Southeast Asia", "South Asia", "West Africa", "North America"]

    mode = rng.choice(modes, n, p=[0.60, 0.20, 0.15, 0.05])
    scheduled = np.array([scheduled_map[m] for m in mode])
    real = np.clip(scheduled + rng.choice([-1, 0, 0, 1, 2], n), 0, None)
    order_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1095, n), unit="D")
    price = np.round(rng.lognormal(4.0, 0.6, n), 2)
    qty = rng.integers(1, 6, n)
    discount = np.round(rng.uniform(0, 0.25, n), 3)

    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "order_date": order_date.strftime("%Y-%m-%d"),
        "shipping_date": (order_date + pd.to_timedelta(real, unit="D")).strftime("%Y-%m-%d"),
        "days_real": real.astype(float),
        "days_scheduled": scheduled,
        "shipping_mode": mode,
        "order_region": rng.choice(regions, n),
        "product_price": price,
        "quantity": qty,
        "discount_rate": discount,
        "sales": np.round(price * qty * (1 - discount), 2),
        "latitude": np.round(rng.uniform(-40, 60, n), 4),
        "longitude": np.round(rng.uniform(-120, 150, n), 4),
        "customer_zipcode": rng.integers(10000, 99999, n).astype(float),
    })

    def pick(k):
        return rng.choice(n, k, replace=False)

    # Inject defects
    for col, frac in [("sales", 0.03), ("days_real", 0.04),
                      ("customer_zipcode", 0.05), ("order_region", 0.02)]:
        df.loc[pick(int(n * frac)), col] = np.nan                      # missing values
    idx = pick(500)
    df.loc[idx[:250], "shipping_mode"] = df.loc[idx[:250], "shipping_mode"].str.lower()
    df.loc[idx[250:], "shipping_mode"] = " " + df.loc[idx[250:], "shipping_mode"] + " "
    df.loc[pick(25), "sales"] *= 30                                    # extreme sales
    df.loc[pick(15), "days_real"] += 60                                # impossible delivery times
    neg = pick(8)
    df.loc[neg, "quantity"] = -df.loc[neg, "quantity"]                 # negative quantity
    df.loc[pick(6), "latitude"] = 999                                  # sentinel value
    bad = pick(12)                                                     # shipped before ordered
    df.loc[bad, "shipping_date"] = (order_date[bad] - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    df = pd.concat([df, df.sample(60, random_state=seed)], ignore_index=True)  # duplicates
    return df


def profile(df):
    """Step 2 - Initial inspection: size, types, missing values, duplicates, skewness."""
    num = df.select_dtypes("number")
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "sales_skew": round(float(num["sales"].skew()), 2) if "sales" in num else None,
    }


def remove_duplicates(df, log):
    """Step 3 - Drop exact duplicates, then duplicate order IDs (keep first)."""
    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="order_id", keep="first")
    log["duplicates_removed"] = before - len(df)
    return df


def standardise(df):
    """Step 4 - Fix inconsistent text and data types."""
    df = df.copy()
    for col in ["shipping_mode", "order_region"]:
        df[col] = df[col].str.strip().str.title()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["shipping_date"] = pd.to_datetime(df["shipping_date"], errors="coerce")
    return df


def validate_rules(df, log):
    """Step 5 - Apply business rules; repair or remove impossible records."""
    df = df.copy()
    neg_qty = df["quantity"] < 0
    df.loc[neg_qty, "quantity"] = df.loc[neg_qty, "quantity"].abs()    # sign-entry error
    log["negative_quantity_fixed"] = int(neg_qty.sum())

    bad_geo = ~df["latitude"].between(-90, 90) | ~df["longitude"].between(-180, 180)
    df.loc[bad_geo, ["latitude", "longitude"]] = np.nan                # treat as missing
    log["invalid_coordinates_nullified"] = int(bad_geo.sum())

    bad_dates = df["shipping_date"] < df["order_date"]                 # shipped before ordered
    log["date_logic_rows_dropped"] = int(bad_dates.sum())
    return df[~bad_dates]


def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def handle_outliers(df, log):
    """Step 6 - Detect with IQR (on log scale for skewed sales) and cap (winsorise)."""
    df = df.copy()
    # Sales are right-skewed, so measure outliers on log1p(sales)
    lo, hi = iqr_bounds(np.log1p(df["sales"]))
    flag = (np.log1p(df["sales"]) > hi) | (np.log1p(df["sales"]) < lo)
    df["sales_outlier_flag"] = flag
    df["sales"] = df["sales"].clip(upper=np.expm1(hi))
    log["sales_outliers_capped"] = int(flag.sum())

    lo, hi = iqr_bounds(df["days_real"])
    flag = (df["days_real"] > hi) | (df["days_real"] < lo)
    df["days_real"] = df["days_real"].clip(lower=max(lo, 0), upper=hi)
    log["delivery_time_outliers_capped"] = int(flag.sum())
    return df


def impute_missing(df, log):
    """Step 7 - Impute using group statistics (numeric) and an explicit label (categorical)."""
    df = df.copy()
    log["missing_before_imputation"] = int(df.isna().sum().sum())
    df["days_real"] = df["days_real"].fillna(df.groupby("shipping_mode")["days_real"].transform("median"))
    df["sales"] = df["sales"].fillna(df.groupby("quantity")["sales"].transform("median"))
    df["latitude"] = df["latitude"].fillna(df["latitude"].median())
    df["longitude"] = df["longitude"].fillna(df["longitude"].median())
    df["customer_zipcode"] = df["customer_zipcode"].fillna(-1)        # -1 = unknown
    df["order_region"] = df["order_region"].fillna("Unknown")
    log["missing_after_imputation"] = int(df.isna().sum().sum())
    return df


def engineer_features(df):
    """Step 8 - Derive analysis-ready features."""
    df = df.copy()
    df["delay_days"] = df["days_real"] - df["days_scheduled"]
    df["is_late"] = (df["delay_days"] > 0).astype(int)
    df["order_month"] = df["order_date"].dt.month
    df["sales_log"] = np.log1p(df["sales"])
    return df


def scale_and_encode(df):
    """Step 9 - Normalise (min-max), standardise (z-score) and one-hot encode."""
    df = df.copy()
    df[["sales_minmax", "days_real_minmax"]] = MinMaxScaler().fit_transform(df[["sales", "days_real"]])
    df[["sales_z", "days_real_z"]] = StandardScaler().fit_transform(df[["sales_log", "days_real"]])
    return pd.get_dummies(df, columns=["shipping_mode", "order_region"], dtype=int)


def final_checks(df):
    """Step 10 - Assert the cleaned data meets the quality rules."""
    assert df.isna().sum().sum() == 0, "missing values remain"
    assert not df.duplicated().any(), "duplicates remain"
    assert (df["quantity"] > 0).all(), "non-positive quantity"
    assert df["latitude"].between(-90, 90).all(), "invalid latitude"
    assert (df["shipping_date"] >= df["order_date"]).all(), "date logic broken"
    return True


def run_pipeline(raw):
    log = {}
    df = remove_duplicates(raw, log)
    df = standardise(df)
    df = validate_rules(df, log)
    df = handle_outliers(df, log)
    df = impute_missing(df, log)
    df = engineer_features(df)
    df = scale_and_encode(df)
    final_checks(df)
    log["rows_final"] = int(len(df))
    return df, log


if __name__ == "__main__":
    raw = generate_sample()
    before = profile(raw)
    clean, log = run_pipeline(raw)
    after = profile(clean)
    print("BEFORE:", json.dumps(before, indent=2))
    print("LOG   :", json.dumps(log, indent=2))
    print("AFTER :", json.dumps(after, indent=2))
    clean.to_csv("logistics_clean.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
        ax[0].boxplot([raw["sales"].dropna(), clean["sales"]], tick_labels=["Raw", "Cleaned"])
        ax[0].set_title("Sales")
        ax[1].boxplot([raw["days_real"].dropna(), clean["days_real"]], tick_labels=["Raw", "Cleaned"])
        ax[1].set_title("Actual shipping days")
        miss = raw.isna().sum()
        miss = miss[miss > 0]
        ax[2].barh(miss.index, miss.values, color="#1F3864")
        ax[2].set_title("Missing values in raw data")
        plt.tight_layout()
        plt.savefig("quality_charts.png", dpi=150)
    except ImportError:
        pass
    json.dump({"before": before, "after": after, "log": log}, open("results.json", "w"))
