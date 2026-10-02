"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
Author : Palarapu Harshavardhan Reddy (Information Technology Department)

Problem : forecast the delivery time (hours) of an order at the moment it is dispatched,
          then use the model's insights to optimise scheduling, driver allocation,
          route sequencing and customer delivery promises.
Data    : a simulated dataset with realistic relationships (see simulate_data()).
          To use real data, pass a CSV path to load_data(); it needs the same columns.
Requires: pandas, numpy, scikit-learn, scipy (matplotlib optional, for charts)
Run     : python logistics_predictive_optimization.py
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

SEED = 42
NUMERIC = ["distance_km", "weight_kg", "traffic_index", "departure_hour",
           "stops_on_route", "driver_experience_years", "is_weekend"]
CATEGORICAL = ["shipping_mode", "weather", "warehouse"]
TARGET = "delivery_time_hours"
OFF_PEAK_HOUR = 13            # departure hour used in the scheduling what-if
COST_PER_VEHICLE_HOUR = 450   # illustrative operating cost (Rs per vehicle-hour)


# ----------------------------------------------------------------------------
# 1. DATA SIMULATION
# ----------------------------------------------------------------------------
def simulate_data(n=6000, seed=SEED):
    """Create a hypothetical delivery dataset with realistic, non-linear effects."""
    rng = np.random.default_rng(seed)
    mode = rng.choice(["Standard", "Express", "Same-Day"], n, p=[0.60, 0.30, 0.10])
    weather = rng.choice(["Clear", "Rain", "Storm"], n, p=[0.70, 0.22, 0.08])
    warehouse = rng.choice(["WH-North", "WH-South", "WH-East", "WH-West"], n)
    distance = np.clip(rng.gamma(2.0, 30.0, n) + 3, 3, 400)
    weight = np.round(rng.lognormal(1.5, 0.8, n), 1)
    hour = rng.integers(6, 21, n)
    weekend = (rng.integers(0, 7, n) >= 5).astype(int)
    peak = ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 19))
    traffic = np.clip(rng.beta(2, 4, n) + 0.30 * peak - 0.10 * weekend, 0.02, 1.0)
    stops = rng.integers(5, 41, n)
    experience = np.clip(rng.exponential(5, n), 0, 20)

    speed = np.select([mode == "Standard", mode == "Express"], [45, 55], 70)
    handling = np.select([mode == "Standard", mode == "Express"], [6, 3], 1)
    weather_mult = np.select([weather == "Clear", weather == "Rain"], [1.0, 1.15], 1.4)
    wh_extra = pd.Series(warehouse).map(
        {"WH-North": 0.0, "WH-South": 0.4, "WH-East": 0.2, "WH-West": 0.6}).values

    travel = distance / speed * (1 + 0.9 * traffic) * weather_mult * (1 - 0.012 * experience)
    true_time = handling + travel + 0.10 * stops + 0.015 * weight + wh_extra
    time = np.maximum(true_time + rng.normal(0, 1, n) * (0.05 * true_time + 0.3), 0.5)

    return pd.DataFrame({
        "order_id": np.arange(1, n + 1), "distance_km": np.round(distance, 1),
        "weight_kg": weight, "shipping_mode": mode, "weather": weather,
        "warehouse": warehouse, "departure_hour": hour, "is_weekend": weekend,
        "traffic_index": np.round(traffic, 3), "stops_on_route": stops,
        "driver_experience_years": np.round(experience, 1),
        TARGET: np.round(time, 2)})


# ----------------------------------------------------------------------------
# 2. DATA LOADING, CLEANING AND PREPARATION
# ----------------------------------------------------------------------------
def load_data(path=None):
    """Read a CSV if a path is given; otherwise use the simulated dataset."""
    if path:
        return pd.read_csv(path)
    return simulate_data()


def clean_data(df):
    """Basic quality safeguards (the full cleaning pipeline is in the Week 2 report)."""
    df = df.drop_duplicates(subset="order_id").dropna(subset=[TARGET])
    df = df[(df[TARGET] > 0) & (df["distance_km"] > 0)].copy()
    df[NUMERIC] = df[NUMERIC].fillna(df[NUMERIC].median())
    for col in CATEGORICAL:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df.reset_index(drop=True)


def prepare_data(df, test_size=0.2):
    """Select features/target and create a reproducible 80/20 train-test split."""
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=SEED)


def make_preprocessor(scale):
    """Scale numeric columns (only needed for linear models) and one-hot encode categories."""
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", numeric, NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])


# ----------------------------------------------------------------------------
# 3. MODEL SELECTION AND TRAINING
# ----------------------------------------------------------------------------
def build_models():
    """Candidate models: one simple baseline, one interpretable, two ensembles."""
    return {
        "Linear Regression": Pipeline([("prep", make_preprocessor(True)),
                                       ("model", LinearRegression())]),
        "Decision Tree": Pipeline([("prep", make_preprocessor(False)),
                                   ("model", DecisionTreeRegressor(
                                       max_depth=6, min_samples_leaf=20, random_state=SEED))]),
        "Random Forest": Pipeline([("prep", make_preprocessor(False)),
                                   ("model", RandomForestRegressor(
                                       n_estimators=200, min_samples_leaf=5,
                                       n_jobs=-1, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("prep", make_preprocessor(False)),
                                       ("model", GradientBoostingRegressor(random_state=SEED))]),
    }


def evaluate(y_true, y_pred):
    """Regression metrics used throughout the report."""
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "MAPE_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100)}


def train_and_test(models, X_train, X_test, y_train, y_test):
    """Fit every candidate on the training set and score it on unseen test data."""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(y_test, model.predict(X_test))
    return results


# ----------------------------------------------------------------------------
# 4. VALIDATION AND TUNING
# ----------------------------------------------------------------------------
def cross_validate_models(models, X_train, y_train, k=5):
    """K-fold cross-validation on the training set to check stability."""
    cv = KFold(n_splits=k, shuffle=True, random_state=SEED)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    out = {}
    for name, model in models.items():
        s = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        out[name] = {"RMSE_mean": float(-s["test_rmse"].mean()), "RMSE_std": float(s["test_rmse"].std()),
                     "MAE_mean": float(-s["test_mae"].mean()), "R2_mean": float(s["test_r2"].mean())}
    return out


def tune_gradient_boosting(X_train, y_train):
    """Randomised hyperparameter search with 3-fold cross-validation."""
    pipe = Pipeline([("prep", make_preprocessor(False)),
                     ("model", GradientBoostingRegressor(random_state=SEED))])
    space = {"model__n_estimators": [100, 200, 300, 400],
             "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
             "model__max_depth": [2, 3, 4, 5],
             "model__min_samples_leaf": [5, 10, 20],
             "model__subsample": [0.7, 0.85, 1.0]}
    search = RandomizedSearchCV(pipe, space, n_iter=15, cv=3, random_state=SEED, n_jobs=-1,
                                scoring="neg_root_mean_squared_error")
    return search.fit(X_train, y_train)


def feature_importance(model, X_test, y_test):
    """Permutation importance: how much RMSE worsens when a feature is shuffled."""
    r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=SEED,
                               scoring="neg_root_mean_squared_error")
    imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False)
    return imp


def residual_analysis(model, X_test, y_test):
    """Check bias, spread and where the model is weakest."""
    resid = y_test - model.predict(X_test)
    by_mode = resid.abs().groupby(X_test["shipping_mode"]).mean()
    return {"mean_residual": float(resid.mean()), "residual_std": float(resid.std()),
            "p90_abs_error": float(resid.abs().quantile(0.9)),
            "MAE_by_shipping_mode": {k: float(v) for k, v in by_mode.items()}}


def promise_window(X_train, y_train, X_test, y_test, point_model, alpha=0.9):
    """Quantile model for delivery promises: how often does the 90th-percentile estimate hold?"""
    q = Pipeline([("prep", make_preprocessor(False)),
                  ("model", GradientBoostingRegressor(loss="quantile", alpha=alpha,
                                                      random_state=SEED, n_estimators=200,
                                                      max_depth=3, learning_rate=0.1))])
    q.fit(X_train, y_train)
    upper = q.predict(X_test)
    return {"target_coverage": alpha, "actual_coverage": float((y_test.values <= upper).mean()),
            "avg_buffer_hours": float((upper - point_model.predict(X_test)).mean())}


# ----------------------------------------------------------------------------
# 5. OPTIMISATION STRATEGIES BASED ON MODEL INSIGHTS
# ----------------------------------------------------------------------------
def is_peak(hour):
    return ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 19))


def shift_departures(model, X_test):
    """Strategy 1 - What-if: move flexible (Standard) peak-hour dispatches to off-peak."""
    flexible = (X_test["shipping_mode"] == "Standard") & is_peak(X_test["departure_hour"])
    base = X_test[flexible]
    shifted = base.copy()
    shifted["departure_hour"] = OFF_PEAK_HOUR
    shifted["traffic_index"] = (shifted["traffic_index"] - 0.30).clip(lower=0.02)  # assumption
    before, after = model.predict(base), model.predict(shifted)
    saved = float((before - after).sum())
    return {"orders_shifted": int(len(base)), "share_of_orders_pct": float(len(base) / len(X_test) * 100),
            "avg_hours_before": float(before.mean()), "avg_hours_after": float(after.mean()),
            "avg_saving_pct": float((before - after).mean() / before.mean() * 100),
            "total_hours_saved": saved, "cost_saved": saved * COST_PER_VEHICLE_HOUR}


def assign_drivers(model, deliveries, driver_experience, n_random=500, seed=SEED):
    """Strategy 2 - Resource allocation: match drivers to deliveries (Hungarian algorithm)."""
    n = len(deliveries)
    cost = np.zeros((n, n))                      # cost[i, j] = predicted hours, delivery i by driver j
    for j in range(n):
        trial = deliveries.copy()
        trial["driver_experience_years"] = driver_experience[j]
        cost[:, j] = model.predict(trial)
    rows, cols = linear_sum_assignment(cost)
    rng = np.random.default_rng(seed)
    random_total = np.mean([cost[np.arange(n), rng.permutation(n)].sum() for _ in range(n_random)])
    optimal = float(cost[rows, cols].sum())
    return {"deliveries": n, "random_total_hours": float(random_total), "optimised_total_hours": optimal,
            "saving_pct": float((random_total - optimal) / random_total * 100),
            "cost_saved": float((random_total - optimal) * COST_PER_VEHICLE_HOUR)}


def travel_time_matrix(coords, speed_kmh=40.0):
    """Congestion-weighted travel times: slower near the city centre (0, 0)."""
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mid = (coords[:, None, :] + coords[None, :, :]) / 2
    congestion = np.exp(-np.linalg.norm(mid, axis=2) / 12.0)
    return dist / speed_kmh * (1 + 0.9 * congestion)


def route_time(route, T):
    """Total time of depot -> stops -> depot (depot is index 0)."""
    path = [0] + list(route) + [0]
    return float(sum(T[a, b] for a, b in zip(path[:-1], path[1:])))


def nearest_neighbour(T):
    unvisited, route, cur = set(range(1, len(T))), [], 0
    while unvisited:
        cur = min(unvisited, key=lambda j: T[cur, j])
        route.append(cur)
        unvisited.remove(cur)
    return route


def two_opt(route, T):
    """Improve a route by reversing segments while that reduces total time."""
    best, improved = route[:], True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if route_time(cand, T) < route_time(best, T) - 1e-9:
                    best, improved = cand, True
    return best


def optimise_route(n_stops=25, seed=SEED):
    """Strategy 3 - Route planning: compare receipt order, nearest-neighbour and 2-opt."""
    rng = np.random.default_rng(seed)
    coords = np.vstack([[-25, -25], rng.uniform(-25, 25, (n_stops, 2))])   # row 0 = depot
    T = travel_time_matrix(coords)
    naive = list(range(1, n_stops + 1))
    nn = nearest_neighbour(T)
    opt = two_opt(nn, T)
    t = {"naive": route_time(naive, T), "nearest_neighbour": route_time(nn, T), "two_opt": route_time(opt, T)}
    return {"stops": n_stops, "hours": t,
            "saving_vs_naive_pct": (t["naive"] - t["two_opt"]) / t["naive"] * 100,
            "cost_saved_per_route": (t["naive"] - t["two_opt"]) * COST_PER_VEHICLE_HOUR,
            "_coords": coords, "_naive": naive, "_opt": opt}


# ----------------------------------------------------------------------------
# 6. FULL RUN
# ----------------------------------------------------------------------------
def main():
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = prepare_data(df)

    models = build_models()
    test_results = train_and_test(models, X_train, X_test, y_train, y_test)
    cv_results = cross_validate_models(build_models(), X_train, y_train)

    search = tune_gradient_boosting(X_train, y_train)
    best = search.best_estimator_
    tuned = evaluate(y_test, best.predict(X_test))
    importance = feature_importance(best, X_test, y_test)
    resid = residual_analysis(best, X_test, y_test)
    promise = promise_window(X_train, y_train, X_test, y_test, best)

    shift = shift_departures(best, X_test)
    rng = np.random.default_rng(SEED)
    batch = X_test.sample(30, random_state=SEED).reset_index(drop=True)
    drivers = np.round(rng.exponential(5, 30).clip(0, 20), 1)
    alloc = assign_drivers(best, batch, drivers)
    route = optimise_route()

    summary = {"n_rows": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
               "target_mean": float(df[TARGET].mean()), "target_std": float(df[TARGET].std()),
               "target_min": float(df[TARGET].min()), "target_max": float(df[TARGET].max()),
               "test_results": test_results, "cv_results": cv_results,
               "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
               "tuned_gb_test": tuned, "importance": importance.round(4).to_dict(),
               "residuals": resid, "promise": promise, "shift": shift, "allocation": alloc,
               "route": {k: v for k, v in route.items() if not k.startswith("_")}}
    return summary, df, best, (X_test, y_test), route


if __name__ == "__main__":
    summary, df, best, (X_test, y_test), route = main()
    t = summary["tuned_gb_test"]
    print("Final model: tuned Gradient Boosting (test set)")
    print("MAE:", round(t["MAE"], 4))
    print("RMSE:", round(t["RMSE"], 4))
    print("R2 Score:", round(t["R2"], 4))
    print(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    df.to_csv("simulated_delivery_dataset.csv", index=False)
    json.dump(summary, open("results4.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        NAVY, ORANGE = "#1F3864", "#E8590C"
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        names = list(summary["test_results"]) + ["GB (tuned)"]
        rmse = [v["RMSE"] for v in summary["test_results"].values()] + [summary["tuned_gb_test"]["RMSE"]]
        ax[0, 0].bar(names, rmse, color=[NAVY] * 4 + [ORANGE])
        ax[0, 0].set_title("Test RMSE by model (hours, lower is better)")
        ax[0, 0].tick_params(axis="x", labelrotation=20)
        pred = best.predict(X_test)
        ax[0, 1].scatter(y_test, pred, s=6, alpha=0.4, color=NAVY)
        lim = [0, max(y_test.max(), pred.max())]
        ax[0, 1].plot(lim, lim, color=ORANGE)
        ax[0, 1].set_xlabel("Actual hours"); ax[0, 1].set_ylabel("Predicted hours")
        ax[0, 1].set_title("Tuned model: predicted vs actual")
        imp = pd.Series(summary["importance"]).sort_values()
        ax[1, 0].barh(imp.index, imp.values, color=NAVY)
        ax[1, 0].set_title("Permutation importance (RMSE increase)")
        ax[1, 1].hist(y_test - pred, bins=40, color=NAVY)
        ax[1, 1].set_title("Residuals (actual - predicted, hours)")
        plt.tight_layout(); plt.savefig("model_charts.png", dpi=140); plt.close()

        c = route["_coords"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        for a, r, title, col in [(ax[0], route["_naive"], "Receipt order", NAVY),
                                 (ax[1], route["_opt"], "Nearest-neighbour + 2-opt", ORANGE)]:
            path = [0] + r + [0]
            a.plot(c[path, 0], c[path, 1], "-o", color=col, ms=4, lw=1)
            a.scatter(*c[0], color="black", s=90, marker="s", zorder=3)
            a.set_title(title); a.set_xlabel("km"); a.set_ylabel("km")
        plt.tight_layout(); plt.savefig("route_charts.png", dpi=140); plt.close()
    except ImportError:
        pass
