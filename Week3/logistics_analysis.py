"""
Week 3: Advanced Data Analysis and Visualization in Logistics
Author: Harshavardhan Reddy
Steps: 1) simulate dataset  2) EDA  3) visualizations  4) export stats
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "bold", "axes.titlesize": 12})
OUT = "charts/"

# ---------------------------------------------------------------- 1. SIMULATION
N = 3000
regions = {"North": 1.0, "South": 1.05, "East": 1.15, "West": 0.95, "Central": 1.0}
modes = {  # base speed km/day, cost per kg-km, reliability
    "Road": dict(speed=450, cpk=0.0009, p=0.50),
    "Rail": dict(speed=380, cpk=0.0006, p=0.20),
    "Air":  dict(speed=1500, cpk=0.0040, p=0.10),
    "Sea":  dict(speed=250, cpk=0.0004, p=0.20),
}
carriers = {"SwiftHaul": 0.0, "BlueLine": 0.4, "TransCo": 0.9, "QuickMove": 0.2}
categories = ["Electronics", "FMCG", "Apparel", "Machinery", "Pharma"]

df = pd.DataFrame({
    "shipment_id": [f"SHP{100000+i}" for i in range(N)],
    "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, N), unit="D"),
    "origin_hub": np.random.choice(["Mumbai", "Delhi", "Chennai", "Kolkata", "Hyderabad"], N),
    "destination_region": np.random.choice(list(regions), N, p=[.22, .22, .16, .24, .16]),
    "transport_mode": np.random.choice(list(modes), N, p=[m["p"] for m in modes.values()]),
    "carrier": np.random.choice(list(carriers), N, p=[.30, .25, .25, .20]),
    "product_category": np.random.choice(categories, N, p=[.2, .3, .2, .15, .15]),
})
df["distance_km"] = np.clip(np.random.gamma(4, 260, N), 80, 4500).round(0)
df["weight_kg"] = np.clip(np.random.lognormal(5.2, 0.9, N), 10, 8000).round(1)
df["shipment_volume_cbm"] = (df.weight_kg / np.random.uniform(120, 250, N)).round(2)
df["month"] = df.order_date.dt.month
df["weekday"] = df.order_date.dt.day_name()

# monthly seasonality (peak in Oct-Dec festive season)
season = df.month.map({10: 1.25, 11: 1.4, 12: 1.35, 1: 0.95, 2: 0.9, 3: 0.95}).fillna(1.0)
# oversample peak months by resampling to create volume trend
w = season / season.sum()
df = df.sample(N, weights=w, replace=True, random_state=1).reset_index(drop=True)
df["shipment_id"] = [f"SHP{100000+i}" for i in range(N)]
season = df.month.map({10: 1.25, 11: 1.4, 12: 1.35, 1: 0.95, 2: 0.9, 3: 0.95}).fillna(1.0)

speed = df.transport_mode.map(lambda m: modes[m]["speed"])
reg = df.destination_region.map(regions)
car = df.carrier.map(carriers)
transit = df.distance_km / speed * reg
handling = np.random.gamma(2, 0.5, N) + (season - 1) * 2.0     # warehouse dwell, peak congestion
df["planned_days"] = np.ceil(transit + 1.5).astype(int)
df["delivery_days"] = np.round(transit + handling + car + np.random.normal(0, 0.5, N), 1).clip(lower=0.5)
df["delay_days"] = (df.delivery_days - df.planned_days).round(1)
df["delayed"] = (df.delay_days > 0).astype(int)

cpk = df.transport_mode.map(lambda m: modes[m]["cpk"])
fuel = 85 + 10 * np.sin(df.month / 12 * 2 * np.pi) + np.random.normal(0, 2, N)   # INR/L index
df["fuel_price_inr"] = fuel.round(1)
df["transport_cost_usd"] = (30 + df.distance_km * df.weight_kg * cpk * (fuel / 90)
                            * np.random.normal(1, 0.08, N) * (1 + 0.15 * (reg - 1))).round(2)
df["cost_per_kg"] = (df.transport_cost_usd / df.weight_kg).round(3)
df["cost_per_km"] = (df.transport_cost_usd / df.distance_km).round(3)

# delay cause for delayed shipments
causes = ["Warehouse congestion", "Carrier capacity", "Weather", "Customs/Docs", "Last-mile failure"]
pc = np.where(df.month.isin([10, 11, 12]), 0.38, 0.22)
df["delay_cause"] = "None"
idx = df.index[df.delayed == 1]
for i in idx:
    p = np.array([pc[i], .25, .15, .12, .0]); p[4] = 1 - p[:4].sum()
    df.at[i, "delay_cause"] = np.random.choice(causes, p=p)

# inject a few missing values & save raw
for c in ["weight_kg", "fuel_price_inr"]:
    df.loc[np.random.choice(df.index, 30, replace=False), c] = np.nan
df.to_csv("logistics_dataset.csv", index=False)
df["weight_kg"] = df.weight_kg.fillna(df.weight_kg.median())
df["fuel_price_inr"] = df.fuel_price_inr.fillna(df.fuel_price_inr.median())

# ---------------------------------------------------------------- 2. EDA
num = ["distance_km", "weight_kg", "shipment_volume_cbm", "planned_days",
       "delivery_days", "delay_days", "transport_cost_usd", "cost_per_kg", "cost_per_km", "fuel_price_inr"]
desc = df[num].describe().T
desc["median"] = df[num].median()
desc["skew"] = df[num].skew()
desc = desc[["mean", "median", "std", "min", "max", "skew"]].round(2)
corr = df[num].corr().round(2)

S = {}
S["n"] = N
S["desc"] = desc.to_dict("index")
S["on_time_rate"] = round(100 * (1 - df.delayed.mean()), 1)
S["avg_delay_when_late"] = round(df.loc[df.delayed == 1, "delay_days"].mean(), 2)
S["corr_dist_cost"] = corr.loc["distance_km", "transport_cost_usd"]
S["corr_weight_cost"] = corr.loc["weight_kg", "transport_cost_usd"]
S["corr_dist_days"] = corr.loc["distance_km", "delivery_days"]
S["corr_fuel_cost"] = corr.loc["fuel_price_inr", "transport_cost_usd"]
S["mode"] = df.groupby("transport_mode").agg(n=("shipment_id", "count"), days=("delivery_days", "mean"),
            cost=("transport_cost_usd", "mean"), cpk=("cost_per_kg", "mean"),
            ontime=("delayed", lambda s: 100 * (1 - s.mean()))).round(2).to_dict("index")
S["carrier"] = df.groupby("carrier").agg(n=("shipment_id", "count"), days=("delivery_days", "mean"),
            ontime=("delayed", lambda s: 100 * (1 - s.mean())), delay=("delay_days", "mean")).round(2).to_dict("index")
S["region"] = df.groupby("destination_region").agg(n=("shipment_id", "count"), days=("delivery_days", "mean"),
            ontime=("delayed", lambda s: 100 * (1 - s.mean()))).round(2).to_dict("index")
monthly = df.groupby("month").agg(n=("shipment_id", "count"), cost=("transport_cost_usd", "mean"),
            ontime=("delayed", lambda s: 100 * (1 - s.mean()))).round(2)
S["monthly"] = monthly.to_dict("index")
S["cause"] = df.loc[df.delayed == 1, "delay_cause"].value_counts().to_dict()
S["peak_ontime"] = round(100 * (1 - df[df.month.isin([10, 11, 12])].delayed.mean()), 1)
S["offpeak_ontime"] = round(100 * (1 - df[~df.month.isin([10, 11, 12])].delayed.mean()), 1)
q1, q3 = df.transport_cost_usd.quantile([.25, .75]); iqr = q3 - q1
S["cost_outliers"] = int((df.transport_cost_usd > q3 + 1.5 * iqr).sum())
S["missing_filled"] = 60
desc.to_csv("summary_statistics.csv"); corr.to_csv("correlation_matrix.csv")

# ---------------------------------------------------------------- 3. VISUALIZATIONS
# Fig 1 histogram + KDE
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df.delivery_days, bins=40, kde=True, color="#2a6f97", ax=ax)
ax.axvline(df.delivery_days.mean(), color="red", ls="--", label=f"Mean {df.delivery_days.mean():.1f}")
ax.axvline(df.delivery_days.median(), color="green", ls="--", label=f"Median {df.delivery_days.median():.1f}")
ax.set(title="Figure 1: Distribution of Delivery Time", xlabel="Delivery time (days)", ylabel="Shipments"); ax.legend()
plt.tight_layout(); plt.savefig(OUT + "fig1_delivery_hist.png"); plt.close()

# Fig 2 boxplot delivery by mode
fig, ax = plt.subplots(figsize=(7, 4))
order = ["Air", "Road", "Rail", "Sea"]
sns.boxplot(data=df, x="transport_mode", y="delivery_days", order=order, ax=ax)
ax.set(title="Figure 2: Delivery Time by Transport Mode", xlabel="Mode", ylabel="Delivery time (days)")
plt.tight_layout(); plt.savefig(OUT + "fig2_box_mode.png"); plt.close()

# Fig 3 monthly trend dual axis
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(monthly.index, monthly.n, color="#a9d6e5", label="Shipments")
ax.set(xlabel="Month (2025)", ylabel="Shipments", title="Figure 3: Monthly Shipment Volume vs On-Time Rate")
ax.set_xticks(range(1, 13))
ax2 = ax.twinx(); ax2.plot(monthly.index, monthly.ontime, color="#d62828", marker="o", label="On-time %")
ax2.set_ylabel("On-time delivery (%)"); ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
plt.tight_layout(); plt.savefig(OUT + "fig3_monthly_trend.png"); plt.close()

# Fig 4 scatter distance vs cost
fig, ax = plt.subplots(figsize=(7, 4.3))
sns.scatterplot(data=df, x="distance_km", y="transport_cost_usd", hue="transport_mode", alpha=.55, s=18, ax=ax)
ax.set_yscale("log")
ax.set(title="Figure 4: Distance vs Transport Cost (log scale)", xlabel="Distance (km)", ylabel="Cost (USD, log)")
plt.tight_layout(); plt.savefig(OUT + "fig4_dist_cost.png"); plt.close()

# Fig 5 correlation heatmap
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7}, ax=ax)
ax.set_title("Figure 5: Correlation Matrix of Numeric Variables")
plt.tight_layout(); plt.savefig(OUT + "fig5_corr.png"); plt.close()

# Fig 6 cost per kg by mode (violin/bar)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=df, x="transport_mode", y="cost_per_kg", order=["Sea", "Rail", "Road", "Air"], errorbar="sd", ax=ax)
ax.set(title="Figure 6: Average Cost per kg by Transport Mode", xlabel="Mode", ylabel="USD per kg")
plt.tight_layout(); plt.savefig(OUT + "fig6_cost_kg.png"); plt.close()

# Fig 7 on-time by carrier
fig, ax = plt.subplots(figsize=(7, 4))
ct = df.groupby("carrier").delayed.apply(lambda s: 100 * (1 - s.mean())).sort_values()
sns.barplot(x=ct.values, y=ct.index, color="#2a9d8f", ax=ax)
for i, v in enumerate(ct.values): ax.text(v + .5, i, f"{v:.1f}%", va="center")
ax.set(title="Figure 7: On-Time Delivery Rate by Carrier", xlabel="On-time (%)", ylabel="", xlim=(0, 100))
plt.tight_layout(); plt.savefig(OUT + "fig7_carrier.png"); plt.close()

# Fig 8 heatmap carrier x month-group delay rate
fig, ax = plt.subplots(figsize=(7, 4))
df["season"] = np.where(df.month.isin([10, 11, 12]), "Peak (Oct-Dec)", "Off-peak (Jan-Sep)")
hm = df.pivot_table(index="carrier", columns="season", values="delayed", aggfunc="mean") * 100
sns.heatmap(hm, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "% delayed"}, ax=ax)
ax.set_title("Figure 8: Delay Rate (%) by Carrier and Season"); ax.set(xlabel="Season", ylabel="Carrier")
plt.tight_layout(); plt.savefig(OUT + "fig8_heatmap.png"); plt.close()
S["heat"] = hm.round(1).to_dict("index")

# Fig 9 Pareto of delay causes
fig, ax = plt.subplots(figsize=(7.5, 4))
cc = df.loc[df.delayed == 1, "delay_cause"].value_counts()
ax.bar(cc.index, cc.values, color="#e76f51")
ax.set_ylabel("Delayed shipments"); ax.tick_params(axis="x", rotation=20)
ax2 = ax.twinx(); ax2.plot(cc.index, 100 * cc.cumsum() / cc.sum(), color="black", marker="o")
ax2.set_ylabel("Cumulative %"); ax2.set_ylim(0, 105); ax2.grid(False)
ax.set_title("Figure 9: Pareto Chart of Delay Causes")
plt.tight_layout(); plt.savefig(OUT + "fig9_pareto.png"); plt.close()
S["cause_pct"] = (100 * cc / cc.sum()).round(1).to_dict()

json.dump(S, open("stats.json", "w"), indent=1, default=float)
print(desc); print(corr[["transport_cost_usd", "delivery_days"]])
print(json.dumps({k: S[k] for k in ["on_time_rate", "avg_delay_when_late", "peak_ontime", "offpeak_ontime",
      "cost_outliers", "mode", "carrier", "region", "cause_pct", "heat"]}, indent=1, default=float))
