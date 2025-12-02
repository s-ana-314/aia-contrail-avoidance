"""Generates a summary json file for the sample flight data."""  # noqa: INP001

from __future__ import annotations

import json

import pandas as pd

from aia_model_contrail_avoidance.airports import list_of_uk_airports

jsonfilename = "2024_01_01_sample_stats"
parquet_file = "flight_data/2024_01_01_sample.parquet"

# Load data
flight_dataframe = pd.read_parquet(parquet_file)

# --- Basic stats ---
timeframe_first = flight_dataframe["timestamp"].min()
timeframe_last = flight_dataframe["timestamp"].max()

number_of_datapoints = len(flight_dataframe)

number_of_flights = flight_dataframe["flight_id"].nunique()

# --- UK / Regional flight classification ---
uk_airports = set(list_of_uk_airports())

# A flight is regional if both departure and arrival airports are UK
regional_flights_df = flight_dataframe[
    (flight_dataframe["departure_airport_icao"].isin(uk_airports))
    & (flight_dataframe["arrival_airport_icao"].isin(uk_airports))
]

number_of_regional_flights = regional_flights_df["flight_id"].nunique()
number_of_international_flights = number_of_flights - number_of_regional_flights

# --- Departure arrival pairs ---
unique_departure_arrival_pairs = (
    flight_dataframe[["departure_airport_icao", "arrival_airport_icao"]].drop_duplicates().shape[0]
)

regional_departure_arrival_pairs = (
    regional_flights_df[["departure_airport_icao", "arrival_airport_icao"]]
    .drop_duplicates()
    .shape[0]
)
# --- Complete flights ---

df = flight_dataframe.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
df["takeoff_time"] = pd.to_datetime(df["takeoff_time"], errors="coerce").dt.tz_localize(None)
df["landing_time"] = pd.to_datetime(df["landing_time"], errors="coerce").dt.tz_localize(None)

complete_flights_df = df[
    (df["takeoff_time"] > timeframe_first) & (df["landing_time"] < timeframe_last)
]

number_of_complete_flights = complete_flights_df["flight_id"].nunique()
# --- Build summary ---
stats = {
    "file_name": parquet_file,
    "number_of_datapoints": number_of_datapoints,
    "timeframe": {"first": str(timeframe_first), "last": str(timeframe_last)},
    "flight_data": {
        "number_of_flights": number_of_flights,
        "number_of_regional_flights": number_of_regional_flights,
        "number_of_international_flights": number_of_international_flights,
    },
    "departure_arrival_pairs": {
        "unique": unique_departure_arrival_pairs,
        "regional": regional_departure_arrival_pairs,
    },
}

# --- Write output ---
with open(f"{jsonfilename}.json", "w") as f:  # noqa: PTH123
    json.dump(stats, f, indent=4)
