"""
Week 1 - Strategic Planning (Logistics)
This file contains basic pseudocode / sample logic
for logistics data analysis.
"""

import pandas as pd

# Load sample dataset
def load_data():
    data = pd.read_csv("sample_logistics_data.csv")
    return data

# Example KPI calculations
def calculate_kpis(data):
    avg_delivery_time = data["delivery_time"].mean()
    on_time_rate = (data["on_time"] == 1).mean() * 100
    
    print("Average Delivery Time:", avg_delivery_time)
    print("On-Time Delivery Rate:", on_time_rate)

# Main execution
if __name__ == "__main__":
    data = load_data()
    calculate_kpis(data)
