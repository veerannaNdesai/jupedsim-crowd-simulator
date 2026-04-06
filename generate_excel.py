import pandas as pd

# Data for Sheet 1: Input Parameters From Client
inputs = [
    {"Category": "General Venue Info", "Parameter Name": "Venue Name", "Description": "Name of the building or site", "Type": "Text", "Example": "Grand Central Mall", "Importance": "Required"},
    {"Category": "General Venue Info", "Parameter Name": "Venue Type", "Description": "Classification of the venue (e.g., MALL, AIRPORT)", "Type": "Text", "Example": "MALL", "Importance": "Required"},
    {"Category": "General Venue Info", "Parameter Name": "Operating Hours", "Description": "Daily open and close hours to simulate", "Type": "Number (Hours)", "Example": "24", "Importance": "Required"},
    
    {"Category": "Spatial Layout", "Parameter Name": "Zone Definitions", "Description": "Semantic names of different areas in the venue", "Type": "List of Texts", "Example": "FoodCourt, Retail_East, Lobby", "Importance": "Required"},
    {"Category": "Spatial Layout", "Parameter Name": "Zone Capacities", "Description": "Max legal or comfortable human capacity per zone", "Type": "List of Numbers", "Example": "FoodCourt: 400, Lobby: 600", "Importance": "Required"},
    {"Category": "Spatial Layout", "Parameter Name": "Available Exits", "Description": "Number of open physical exits", "Type": "Number", "Example": "4", "Importance": "Required"},
    {"Category": "Spatial Layout", "Parameter Name": "Zone Popularity (Attractors)", "Description": "1-10 ranking of how likely crowds are to visit each zone", "Type": "List of Numbers", "Example": "FoodCourt: 8, Lobby: 10, Pathways: 4", "Importance": "Required"},

    {"Category": "Audience Profiling", "Parameter Name": "Crowd Demographics", "Description": "Types of people in the venue and their percentage", "Type": "Percentages", "Example": "50% Shoppers, 30% Visitors, 20% Staff", "Importance": "Required"},
    {"Category": "Audience Profiling", "Parameter Name": "Dwell Times", "Description": "Average minutes each group spends in the venue", "Type": "Minutes", "Example": "Shoppers: 60 min, Staff: 480 min", "Importance": "Required"},
    {"Category": "Audience Profiling", "Parameter Name": "Walking Speeds", "Description": "Average walking speed per group (can use defaults)", "Type": "Meters / Second", "Example": "1.2 m/s", "Importance": "Optional"},

    {"Category": "Traffic Dynamics", "Parameter Name": "Peak Daily Footfall", "Description": "Max number of people in the building at the busiest time", "Type": "Number", "Example": "3200", "Importance": "Required"},
    {"Category": "Traffic Dynamics", "Parameter Name": "Hourly Traffic Curve", "Description": "Percentage of peak footfall per hour of the day (0-23)", "Type": "List of %", "Example": "9AM: 10%, 1PM: 100%, 8PM: 60%", "Importance": "Required"},

    {"Category": "Special Events", "Parameter Name": "Event Type", "Description": "Type of event occurring (if any)", "Type": "Text", "Example": "Flash Sale, Movie Premiere", "Importance": "Optional"},
    {"Category": "Special Events", "Parameter Name": "Event Location & Time", "Description": "Zone affected and Start/End hours", "Type": "Text / Hours", "Example": "Zone: EventArea, Hours: 14 to 18", "Importance": "Optional"},
    {"Category": "Special Events", "Parameter Name": "Crowd Spike Multiplier", "Description": "How much footfall multiplies during the event", "Type": "Decimal", "Example": "2.5x", "Importance": "Optional"},

    {"Category": "Anomalies / Environment", "Parameter Name": "Blocked Exits", "Description": "Names of any exits closed for maintenance/emergency", "Type": "List of Texts", "Example": "Exit_North", "Importance": "Optional"},
    {"Category": "Anomalies / Environment", "Parameter Name": "Weather & Temp", "Description": "Weather conditions during the day", "Type": "Text & Number", "Example": "Sunny, 28C", "Importance": "Optional"},
]

# Data for Sheet 2: Output Features Provided to Client
outputs = [
    {"Category": "Identifiers", "Feature Name": "Zone_name", "Description": "Name of the specific spatial area", "Example": "FoodCourt, Lobby", "ML Value": "Groups time-series data spatially"},
    {"Category": "Identifiers", "Feature Name": "hour_of_day", "Description": "The simulated hour (0-23)", "Example": "14", "ML Value": "Captures daily seasonality and trends"},
    {"Category": "Identifiers", "Feature Name": "day_of_week", "Description": "0=Monday, 6=Sunday", "Example": "5", "ML Value": "Captures weekly seasonality"},
    
    {"Category": "Crowd Density & Volume", "Feature Name": "count", "Description": "Absolute number of people in the zone at that hour", "Example": "150", "ML Value": "Primary target variable for forecasting"},
    {"Category": "Crowd Density & Volume", "Feature Name": "density", "Description": "People per square meter", "Example": "0.45", "ML Value": "Indicates physical spacing / safety risk"},
    {"Category": "Crowd Density & Volume", "Feature Name": "occupancy_ratio", "Description": "Current count divided by max zone capacity", "Example": "0.85 (85% full)", "ML Value": "Normalized metric for cross-zone comparison"},
    
    {"Category": "Movement & Flow", "Feature Name": "avg_speed", "Description": "Average walking speed of agents in the zone", "Example": "0.9 m/s", "ML Value": "Drops significantly during severe congestion"},
    {"Category": "Movement & Flow", "Feature Name": "flow_in", "Description": "Number of new people entering the zone that hour", "Example": "45", "ML Value": "Predicts future crowding"},
    {"Category": "Movement & Flow", "Feature Name": "flow_out", "Description": "Number of people leaving the zone that hour", "Example": "30", "ML Value": "Predicts zone clearance rate"},
    {"Category": "Movement & Flow", "Feature Name": "flow_imbalance", "Description": "flow_in minus flow_out", "Example": "15", "ML Value": "Positive value indicates crowd accumulation"},
    {"Category": "Movement & Flow", "Feature Name": "avg_dwelltime", "Description": "Average time spent by agents currently in zone", "Example": "45.5 mins", "ML Value": "Indicates if a zone is a transient path or a destination"},
    
    {"Category": "Historical / Lag (Time-Series)", "Feature Name": "count_t-1", "Description": "Count of people in the previous hour", "Example": "120", "ML Value": "Autoregressive feature for ML models"},
    {"Category": "Historical / Lag (Time-Series)", "Feature Name": "count_trend", "Description": "Difference in count from previous hour", "Example": "+30", "ML Value": "Velocity of crowd buildup"},
    {"Category": "Historical / Lag (Time-Series)", "Feature Name": "density_change_rate", "Description": "Percentage change in density from previous hour", "Example": "0.25 (25% increase)", "ML Value": "Identifies rapid sudden surges"},
    {"Category": "Historical / Lag (Time-Series)", "Feature Name": "rolling_count_3h", "Description": "Moving average of count over last 3 hours", "Example": "110", "ML Value": "Smoothes out short-term noise"},
    
    {"Category": "Spatial Network (Graph)", "Feature Name": "neighbor_density_avg", "Description": "Average density of physically connected adjacent zones", "Example": "0.60", "ML Value": "Predicts spillback congestion from nearby areas"},
    {"Category": "Spatial Network (Graph)", "Feature Name": "neighbor_flow_in", "Description": "Total flow entering adjacent zones", "Example": "80", "ML Value": "Early warning for upcoming waves of people"},
    
    {"Category": "Risk & Safety", "Feature Name": "pressure_score", "Description": "Composite metric (density × speed) indicating crowd tension", "Example": "0.55", "ML Value": "Key indicator for safety incidents/stampede risk"},
    {"Category": "Risk & Safety", "Feature Name": "exit_blocked_flag", "Description": "Boolean (0 or 1) if an exit is locally blocked", "Example": "1", "ML Value": "Models non-standard emergency conditions"},
    
    {"Category": "Contextual & External", "Feature Name": "event_type / scale", "Description": "Type and magnitude of ongoing special event", "Example": "Concert / Large", "ML Value": "Categorical trigger for anomalous behavior"},
    {"Category": "Contextual & External", "Feature Name": "time_to_event_start", "Description": "Hours remaining until a scheduled event starts", "Example": "2", "ML Value": "Models arrival build-up patterns"},
    {"Category": "Contextual & External", "Feature Name": "weather / temperature_c", "Description": "External weather conditions", "Example": "Rain / 15", "ML Value": "Models weather impact on indoor attendance"}
]

df_in = pd.DataFrame(inputs)
df_out = pd.DataFrame(outputs)

excel_path = "c:\\Users\\veeranna.n\\Desktop\\jupedsim_ml_pipeline_full\\Client_Data_Specification.xlsx"
import os
try:
    with pd.ExcelWriter(excel_path) as writer:
        df_in.to_excel(writer, sheet_name="Required Inputs (From Client)", index=False)
        df_out.to_excel(writer, sheet_name="Model Outputs (To Client)", index=False)
    print(f"Excel file created successfully at {excel_path}")
except ModuleNotFoundError:
    print("openpyxl is required to write excel files. Installing...")
    os.system("..\\env314\\Scripts\\python.exe -m pip install openpyxl")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_in.to_excel(writer, sheet_name="Required Inputs (From Client)", index=False)
        df_out.to_excel(writer, sheet_name="Model Outputs (To Client)", index=False)
    print(f"Excel file created successfully at {excel_path}")
