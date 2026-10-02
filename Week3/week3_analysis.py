"""
Week 3 – Logistics Data Analysis & Visualization

This script performs:
- Data cleaning
- KPI computation
- Exploratory Data Analysis (EDA)
- Visualization of logistics performance

Libraries used: pandas, numpy, matplotlib
"""
Requires: pandas, numpy, matplotlib, scikit-learn, ortools
"""
import pandas as pd
import numpy as np

def load_data(orders_path, inventory_path):
    """Read raw order and inventory extracts."""
    orders = pd.read_csv(orders_path,
                         parse_dates=["order_date", "promised_date", "delivery_date"])
    inventory = pd.read_csv(inventory_path, parse_dates=["snapshot_date"])
    return orders, inventory

def clean_orders(orders):
    """Remove duplicates, missing keys and impossible dates; add derived fields."""
    orders = orders.drop_duplicates(subset="order_id")
    orders = orders.dropna(subset=["order_date", "delivery_date", "warehouse_id"])
    orders = orders[orders["delivery_date"] >= orders["order_date"]].copy()
    orders["lead_time_days"] = (orders["delivery_date"] - orders["order_date"]).dt.days
    orders["is_on_time"] = orders["delivery_date"] <= orders["promised_date"]
    return orders


def compute_kpis(orders, inventory):
    """Return the core logistics KPIs as a dictionary."""
    on_time_rate = orders["is_on_time"].mean() * 100
    avg_lead_time = orders["lead_time_days"].mean()
    cost_per_delivery = orders["shipping_cost"].sum() / len(orders)
    inventory_turnover = orders["cogs"].sum() / inventory["inventory_value"].mean()
    stockout_rate = (inventory["stock_on_hand"] == 0).mean() * 100
    return {
        "On-Time Delivery %": round(on_time_rate, 2),
        "Avg Lead Time (days)": round(avg_lead_time, 2),
        "Cost per Delivery": round(cost_per_delivery, 2),
        "Inventory Turnover": round(inventory_turnover, 2),
        "Stockout Rate %": round(stockout_rate, 2),
    }


import matplotlib.pyplot as plt

def run_eda(orders):
    """Exploratory analysis: distributions, trends, correlations."""
    print(orders.describe(include="all"))

    # Demand trend and seasonality
    monthly = orders.set_index("order_date").resample("M")["quantity"].sum()
    monthly.plot(title="Monthly Demand (units)")
    plt.show()

    # Service level by warehouse
    (orders.groupby("warehouse_id")["is_on_time"].mean() * 100) \
        .sort_values().plot(kind="barh", title="On-Time Delivery % by Warehouse")
    plt.show()

    # Which factors move with lead time and cost?
    cols = ["lead_time_days", "distance_km", "shipping_cost", "quantity"]
    print(orders[cols].corr())


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

def build_features(daily):
    """daily: DataFrame indexed by date with a 'demand' column."""
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["lag_7"] = daily["demand"].shift(7)
    daily["lag_28"] = daily["demand"].shift(28)
    daily["rolling_mean_7"] = daily["demand"].shift(1).rolling(7).mean()
    return daily.dropna()

def evaluate_forecast(daily):
    """Time-series cross-validation; returns average MAPE (%)."""
    data = build_features(daily.copy())
    X, y = data.drop(columns="demand"), data["demand"]
    errors = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        errors.append(mean_absolute_percentage_error(y.iloc[test_idx], preds))
    return np.mean(errors) * 100


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster_delivery_zones(orders, k_range=range(3, 9)):
    """Group delivery destinations into zones to design efficient routes."""
    coords = orders[["dest_lat", "dest_lon"]].dropna()
    best_k, best_score = None, -1
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(coords)
        score = silhouette_score(coords, labels)
        if score > best_score:
            best_k, best_score = k, score
    final = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(coords)
    return final.labels_, final.cluster_centers_, best_k


from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_vrp(distance_matrix, demands, vehicle_capacity, num_vehicles, depot=0):
    """Capacitated Vehicle Routing Problem solved with Google OR-Tools."""
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(i, j):
        return int(distance_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)])
    transit = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    def demand_cb(i):
        return demands[manager.IndexToNode(i)]
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [vehicle_capacity] * num_vehicles, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 30

    solution = routing.SolveWithParameters(params)
    routes = []
    if solution:
        for v in range(num_vehicles):
            index, route = routing.Start(v), []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            routes.append(route)
    return routes


if __name__ == "__main__":
    orders, inventory = load_data("orders.csv", "inventory.csv")
    orders = clean_orders(orders)
    print(compute_kpis(orders, inventory))
    run_eda(orders)
