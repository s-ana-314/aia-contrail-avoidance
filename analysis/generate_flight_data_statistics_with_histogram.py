"""Generates a summary json file for the sample flight data, including a histogram of distance traveled in segment."""  # noqa: INP001

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from aia_model_contrail_avoidance.core_model.airports import list_of_uk_airports
from aia_model_contrail_avoidance.core_model.dimensions import (
    TemporalGranularity,
    _get_temporal_grouping_field,
    _get_temporal_range_and_labels,
    user_input_temporal_granularity,
)

DISTANCE_BINS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 2000, 3000]
ALTITUDE_BIN_SIZE = 10  # in flight levels (1000 ft)
TEMPORAL_GRANULARITY = user_input_temporal_granularity()  # Granularity for temporal aggregation


def _create_histogram(data: pl.Series, bins: list[float]) -> dict[str, list[float] | float]:
    """Create histogram from data and bins."""
    hist, bin_edges = np.histogram(data.drop_nulls().to_numpy(), bins=bins)
    return {"bin_edges": bin_edges.tolist(), "counts": hist.tolist()}


def generate_flight_statistics(  # noqa: PLR0915
    parquet_file_name: str,
    jsonfilename: str,
    temporal_granularity: TemporalGranularity = TemporalGranularity.HOURLY,
) -> None:
    """Generate flight data statistics and save to JSON file.

    Args:
        parquet_file_name: Path to the parquet file containing flight data.
        jsonfilename: Output JSON filename (without extension).
        temporal_granularity: Temporal granularity for aggregation (default: HOURLY).
    """
    # Load data
    parquet_file_path = Path("data/contrails_model_data/" + parquet_file_name + ".parquet")
    if not parquet_file_path.exists():
        msg = f"Parquet file not found: data/contrails_model_data/{parquet_file_name}.parquet"
        raise FileNotFoundError(msg)

    flight_dataframe = pl.read_parquet(parquet_file_path)

    # --- Basic stats ---
    timeframe_first = flight_dataframe["timestamp"].min()
    timeframe_last = flight_dataframe["timestamp"].max()
    number_of_datapoints = len(flight_dataframe)
    number_of_flights = flight_dataframe["flight_id"].n_unique()

    # --- UK / Regional flight classification ---
    uk_airports = set(list_of_uk_airports())
    regional_flights_df = flight_dataframe.filter(
        (pl.col("departure_airport_icao").is_in(uk_airports))
        & (pl.col("arrival_airport_icao").is_in(uk_airports))
    )
    number_of_regional_flights = regional_flights_df["flight_id"].n_unique()
    number_of_international_flights = number_of_flights - number_of_regional_flights

    # --- Departure arrival pairs ---
    unique_departure_arrival_pairs = (
        flight_dataframe.select(["departure_airport_icao", "arrival_airport_icao"]).unique().height
    )
    regional_departure_arrival_pairs = (
        regional_flights_df.select(["departure_airport_icao", "arrival_airport_icao"])
        .unique()
        .height
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
            flight_dataframe[altitude_col].null_count() / len(flight_dataframe) * 100
        )

        min_val = flight_dataframe[altitude_col].cast(pl.Float64).min()
        max_val = flight_dataframe[altitude_col].cast(pl.Float64).max()

        if min_val is not None and max_val is not None:
            min_int: int = int(min_val)  # type: ignore[arg-type]
            max_int: int = int(max_val)  # type: ignore[arg-type]
            bins = [float(x) for x in range(min_int, max_int + 10, 10)]
        else:
            bins = []
        altitude_histogram = _create_histogram(flight_dataframe[altitude_col], bins)
        altitude_histogram["empty_percentage"] = round(altitude_empty_percentage, 2)

    # --- Distance flown at each altitude band ---
    if altitude_col in flight_dataframe.columns and distance_col in flight_dataframe.columns:
        distance_by_altitude_bin = []
        valid_altitude_df = flight_dataframe.filter(pl.col(altitude_col).is_not_null())
        altitude_values = valid_altitude_df[altitude_col].to_numpy()

        for i in range(1, len(bins)):
            bin_indices = np.digitize(altitude_values, bins) == i
            if bin_indices.any():
                filtered_df = valid_altitude_df.filter(pl.Series(bin_indices))
                total_distance = filtered_df[distance_col].sum()
            else:
                total_distance = 0.0
            distance_by_altitude_bin.append(float(total_distance))

        distance_by_altitude_histogram = {
            "bin_edges": bins,
            "distance_flown": distance_by_altitude_bin,
        }

    # --- Number of Aircraft per temporal unit ---
    planes_per_temporal_histogram = None
    if "timestamp" in flight_dataframe.columns:
        temporal_field = _get_temporal_grouping_field(temporal_granularity)
        temporal_range, _ = _get_temporal_range_and_labels(temporal_granularity)

        # Extract temporal unit from timestamp
        flight_dataframe_with_temporal = flight_dataframe.with_columns(
            pl.col("timestamp").dt.__getattribute__(temporal_field)().alias("temporal_unit")
        )
        planes_per_temporal = (
            flight_dataframe_with_temporal.group_by("temporal_unit")
            .agg(pl.col("flight_id").n_unique())
            .to_dict(as_series=False)
        )
        temporal_to_planes = dict(
            zip(planes_per_temporal["temporal_unit"], planes_per_temporal["flight_id"], strict=True)
        )
        planes_per_temporal_histogram = {
            str(unit): temporal_to_planes.get(unit, 0) for unit in temporal_range
        }

    distance_flown_per_temporal_histogram = None
    if "timestamp" in flight_dataframe.columns and distance_col in flight_dataframe.columns:
        temporal_field = _get_temporal_grouping_field(temporal_granularity)
        temporal_range, _ = _get_temporal_range_and_labels(temporal_granularity)

        # Extract temporal unit from timestamp
        flight_dataframe_with_temporal = flight_dataframe.with_columns(
            pl.col("timestamp").dt.__getattribute__(temporal_field)().alias("temporal_unit")
        )
        distance_per_temporal = (
            flight_dataframe_with_temporal.group_by("temporal_unit")
            .agg(pl.col(distance_col).sum())
            .to_dict(as_series=False)
        )
        temporal_to_distance = dict(
            zip(
                distance_per_temporal["temporal_unit"],
                distance_per_temporal[distance_col],
                strict=True,
            )
        )
        distance_flown_per_temporal_histogram = {
            str(unit): temporal_to_distance.get(unit, 0) for unit in temporal_range
        }

    # --- Build summary ---
    stats = {
        "file_name": parquet_file_name,
        "temporal_granularity": temporal_granularity.value,
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
        "air_traffic_density_per_temporal_histogram": planes_per_temporal_histogram,
        "distance_flown_per_temporal_histogram": distance_flown_per_temporal_histogram,
    }

    # --- Write output ---
    with open(f"results/{jsonfilename}.json", "w") as f:  # noqa: PTH123
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    generate_flight_statistics(
        parquet_file_name="2024_01_01_sample_processed_with_interpolation",
        jsonfilename="2024_01_01_sample_stats_processed",
    )
