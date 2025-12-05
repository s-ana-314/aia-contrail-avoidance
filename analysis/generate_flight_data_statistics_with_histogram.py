"""Generates a summary json file for the sample flight data, including a histogram of distance traveled in segment."""  # noqa: INP001

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aia_model_contrail_avoidance.airports import list_of_uk_airports

DISTANCE_BINS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 2000, 3000]
ALTITUDE_BIN_SIZE = 10  # in flight levels (1000 ft)


def _create_histogram(data: pd.Series, bins: list[float]) -> dict:
    """Create histogram from data and bins."""
    hist, bin_edges = np.histogram(data.dropna(), bins=bins)
    return {"bin_edges": bin_edges.tolist(), "counts": hist.tolist()}


def generate_flight_statistics(parquet_file_name: str, jsonfilename: str) -> None:
    """Generate flight data statistics and save to JSON file.

    Args:
        parquet_file_name: Path to the parquet file containing flight data.
        jsonfilename: Output JSON filename (without extension).
    """
    # Load data
    parquet_file_path = Path("data/" + parquet_file_name + ".parquet")
    if not parquet_file_path.exists():
        msg = f"Parquet file not found: data/{parquet_file_name}.parquet"
        raise FileNotFoundError(msg)

    flight_dataframe = pd.read_parquet(parquet_file_path)

    # --- Basic stats ---
    timeframe_first = flight_dataframe["timestamp"].min()
    timeframe_last = flight_dataframe["timestamp"].max()
    number_of_datapoints = len(flight_dataframe)
    number_of_flights = flight_dataframe["flight_id"].nunique()

    # --- UK / Regional flight classification ---
    uk_airports = set(list_of_uk_airports())
    regional_flights_df = flight_dataframe[
        (flight_dataframe["departure_airport_icao"].isin(uk_airports))
        & (flight_dataframe["arrival_airport_icao"].isin(uk_airports))
    ]
    number_of_regional_flights = regional_flights_df["flight_id"].nunique()
    number_of_international_flights = number_of_flights - number_of_regional_flights

    # --- Departure arrival pairs ---
    unique_departure_arrival_pairs = (
        flight_dataframe[["departure_airport_icao", "arrival_airport_icao"]]
        .drop_duplicates()
        .shape[0]
    )
    regional_departure_arrival_pairs = (
        regional_flights_df[["departure_airport_icao", "arrival_airport_icao"]]
        .drop_duplicates()
        .shape[0]
    )

    # --- Histogram of distance flown in segment ---
    distance_col = "distance_flown_in_segment"
    histogram = (
        _create_histogram(flight_dataframe[distance_col], DISTANCE_BINS)
        if distance_col in flight_dataframe.columns
        else None
    )

    # --- Histogram of altitude ---
    altitude_histogram = None
    distance_by_altitude_histogram = None
    altitude_col = "flight_level"

    if altitude_col in flight_dataframe.columns:
        altitude_empty_percentage = (
            flight_dataframe[altitude_col].isna().sum() / len(flight_dataframe) * 100
        )
        min_val = flight_dataframe[altitude_col].min()
        max_val = flight_dataframe[altitude_col].max()

        bins = list(range(int(min_val), int(max_val) + 10, 10))
        altitude_histogram = _create_histogram(flight_dataframe[altitude_col], bins)
        altitude_histogram["empty_percentage"] = round(altitude_empty_percentage, 2)

    # --- Distance flown at each altitude band ---
    if altitude_col in flight_dataframe.columns and distance_col in flight_dataframe.columns:
        distance_by_altitude_bin = []
        valid_altitude_mask = flight_dataframe[altitude_col].notna()

        for i in range(1, len(bins)):
            mask = valid_altitude_mask & (np.digitize(flight_dataframe[altitude_col], bins) == i)
            total_distance = flight_dataframe.loc[mask, distance_col].sum()
            distance_by_altitude_bin.append(float(total_distance))

        distance_by_altitude_histogram = {
            "bin_edges": bins,
            "distance_flown": distance_by_altitude_bin,
        }

    # --- Number of Aircraft per hour ---
    planes_per_hour_histogram = None
    if "timestamp" in flight_dataframe.columns:
        # Extract hour from timestamp
        flight_dataframe_copy = flight_dataframe.copy()
        flight_dataframe_copy["hour"] = pd.to_datetime(flight_dataframe_copy["timestamp"]).dt.hour
        planes_per_hour = flight_dataframe_copy.groupby("hour")["flight_id"].nunique().to_dict()
        planes_per_hour_histogram = {str(hour): planes_per_hour.get(hour, 0) for hour in range(24)}

    # --- Build summary ---
    stats = {
        "file_name": parquet_file_name,
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
        "distance_flown_in_segment_histogram": histogram,
        "altitude_baro_histogram": altitude_histogram,
        "distance_flown_by_altitude_histogram": distance_by_altitude_histogram,
        "planes_per_hour_histogram": planes_per_hour_histogram,
    }

    # --- Write output ---
    with open(f"results/{jsonfilename}.json", "w") as f:  # noqa: PTH123
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    generate_flight_statistics(
        parquet_file_name="2024_01_01_sample_processed",
        jsonfilename="2024_01_01_sample_stats_processed",
    )
