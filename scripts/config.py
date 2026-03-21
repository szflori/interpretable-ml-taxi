# Raw -> normalized column naming used in notebooks.
DEFAULT_RENAME_MAP: dict[str, str] = {
    "Trip_Pickup_DateTime": "pickup_at",
    "Trip_Dropoff_DateTime": "dropoff_at",
    "Passenger_Count": "passenger_count",
    "Trip_Distance": "trip_distance",
    "Start_Lon": "pickup_lon",
    "Start_Lat": "pickup_lat",
    "End_Lon": "dropoff_lon",
    "End_Lat": "dropoff_lat",
}

# Raw columns we usually keep for EDA + baseline feature engineering.
DEFAULT_COLUMNS_TO_KEEP: list[str] = list(DEFAULT_RENAME_MAP.keys())

DEFAULT_NUMERIC_COLUMNS: list[str] = [
    "passenger_count",
    "trip_distance",
    "pickup_lon",
    "pickup_lat",
    "dropoff_lon",
    "dropoff_lat",
]

DEFAULT_CLEAN_COLUMNS: list[str] = list(DEFAULT_RENAME_MAP.values())
