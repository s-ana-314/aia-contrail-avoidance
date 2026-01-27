"""Generate energy forcing summary statistics including contrail formation analysis."""  # noqa: INP001

from __future__ import annotations

import json

import numpy as np
import polars as pl

from aia_model_contrail_avoidance.core_model.airports import list_of_uk_airports
from aia_model_contrail_avoidance.core_model.dimensions import (
    TemporalGranularity,
    _get_temporal_grouping_field,
    _get_temporal_range_and_labels,
)


def generate_energy_forcing_statistics(  # noqa: PLR0915
    parquet_file: str,
    output_filename: str | None = None,
    temporal_granularity: TemporalGranularity = TemporalGranularity.HOURLY,
) -> None:
    """Generate energy forcing summary statistics including contrail formation analysis.

    Args:
        parquet_file: Path to the parquet file containing flight data with energy forcing.
        output_filename: Optional path to save the statistics as JSON. If None, no file is written.
        temporal_granularity: Temporal granularity for aggregation (default: HOURLY).

    """
    # Load the flight data with energy forcing
    flight_dataframe = pl.read_parquet(f"data/contrails_model_data/{parquet_file}.parquet")

    # --- Basic Energy Forcing Statistics ---
    total_datapoints = len(flight_dataframe)
    total_flights = flight_dataframe["flight_id"].n_unique()

    # Calculate total distance flown
    total_distance_flown = flight_dataframe["distance_flown_in_segment"].sum()

    # -- Airspace Specific Statistics ---
    uk_airspace_segments = flight_dataframe.filter(pl.col("airspace").is_not_null())
    total_energy_forcing_in_uk_airspace = uk_airspace_segments["ef"].sum()

    international_airspace_segments = flight_dataframe.filter(pl.col("airspace").is_null())
    total_energy_forcing_in_international_airspace = international_airspace_segments["ef"].sum()

    # Regional vs International Flights
    uk_airports = list_of_uk_airports()
    regional_flights_df = flight_dataframe.filter(
        pl.col("arrival_airport_icao").is_in(uk_airports)
        & pl.col("departure_airport_icao").is_in(uk_airports)
    )
    number_of_regional_flights = regional_flights_df["flight_id"].n_unique()
    number_of_international_flights = total_flights - number_of_regional_flights
    # --- Contrail Formation Analysis ---

    # Segments with positive energy forcing are forming contrails
    contrail_forming_segments = flight_dataframe.filter(pl.col("ef") > 0)

    # Distance that forms contrails
    distance_forming_contrails = contrail_forming_segments["distance_flown_in_segment"].sum()
    percentage_distance_forming_contrails = (
        (distance_forming_contrails / total_distance_flown) * 100 if total_distance_flown > 0 else 0
    )

    # Number of flights that form contrails (at least one segment with ef > 0)
    flights_forming_contrails = contrail_forming_segments["flight_id"].n_unique()
    percentage_flights_forming_contrails = (flights_forming_contrails / total_flights) * 100

    # --- Energy Forcing Statistics ---
    total_energy_forcing = flight_dataframe["ef"].sum()
    mean_energy_forcing_per_segment = flight_dataframe["ef"].mean()
    median_energy_forcing_per_segment = flight_dataframe["ef"].median()
    max_energy_forcing_per_segment = flight_dataframe["ef"].max()

    # Energy forcing statistics for contrail-forming segments only
    mean_ef_contrail_segments = contrail_forming_segments["ef"].mean()
    median_ef_contrail_segments = contrail_forming_segments["ef"].median()

    # Per-flight energy forcing statistics
    flight_ef_summary = flight_dataframe.group_by("flight_id").agg(
        pl.col("ef").sum().alias("total_ef")
    )

    # generate a per flight histogram of ef values as a cumualative histogram
    ef_values = flight_ef_summary["total_ef"].to_numpy().astype("float64")
    hist_counts, bin_edges = np.histogram(ef_values, bins=50, density=False)
    cumulative_counts = hist_counts.cumsum()
    ef_histogram = {
        "bin_edges": bin_edges.tolist(),
        "counts": hist_counts.tolist(),
        "cumulative_counts": cumulative_counts.tolist(),
    }
    # distance flown per temporal unit
    distance_flown_per_temporal_histogram = None
    if (
        "timestamp" in flight_dataframe.columns
        and "distance_flown_in_segment" in flight_dataframe.columns
    ):
        temporal_field = _get_temporal_grouping_field(temporal_granularity)
        temporal_range, _labels = _get_temporal_range_and_labels(temporal_granularity)

        flight_dataframe_with_temporal = flight_dataframe.with_columns(
            pl.col("timestamp").dt.__getattribute__(temporal_field)().alias("temporal_unit")
        )
        distance_per_temporal = (
            flight_dataframe_with_temporal.group_by("temporal_unit")
            .agg(pl.col("distance_flown_in_segment").sum())
            .to_dict(as_series=False)
        )
        temporal_to_distance = dict(
            zip(
                distance_per_temporal["temporal_unit"],
                distance_per_temporal["distance_flown_in_segment"],
                strict=True,
            )
        )
        distance_flown_per_temporal_histogram = {
            str(unit): temporal_to_distance.get(unit, 0) for unit in temporal_range
        }
    # histogram of distance forming contrails per temporal unit (Default: HOURLY)
    distance_forming_contrails_per_temporal_histogram = None
    if "timestamp" in contrail_forming_segments.columns:
        temporal_field = _get_temporal_grouping_field(temporal_granularity)
        temporal_range, _labels = _get_temporal_range_and_labels(temporal_granularity)

        contrail_forming_segments_with_temporal = contrail_forming_segments.with_columns(
            pl.col("timestamp").dt.__getattribute__(temporal_field)().alias("temporal_unit")
        )

        distance_per_temporal = (
            contrail_forming_segments_with_temporal.group_by("temporal_unit")
            .agg(pl.col("distance_flown_in_segment").sum())
            .to_dict(as_series=False)
        )

        temporal_to_distance = dict(
            zip(
                distance_per_temporal["temporal_unit"],
                distance_per_temporal["distance_flown_in_segment"],
                strict=True,
            )
        )

        # Ensure all temporal units are present (fill missing with 0)
        distance_forming_contrails_per_temporal_histogram = {
            str(unit): temporal_to_distance.get(unit, 0) for unit in temporal_range
        }
    else:
        distance_forming_contrails_per_temporal_histogram = None

    # Air traffic density per temporal unit
    # (number of unique flights per temporal unit)
    air_traffic_density_per_temporal_histogram = None
    if "timestamp" in flight_dataframe.columns:
        temporal_field = _get_temporal_grouping_field(temporal_granularity)
        temporal_range, _labels = _get_temporal_range_and_labels(temporal_granularity)

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
        air_traffic_density_per_temporal_histogram = {
            str(unit): temporal_to_planes.get(unit, 0) for unit in temporal_range
        }
    else:
        air_traffic_density_per_temporal_histogram = None

    # --- Build Summary ---
    stats = {
        "file_name": parquet_file,
        "overview": {
            "total_datapoints": total_datapoints,
        },
        "contrail_formation": {
            "flights_forming_contrails": int(flights_forming_contrails),
            "percentage_flights_forming_contrails": round(percentage_flights_forming_contrails, 2),
            "distance_forming_contrails_nm": float(distance_forming_contrails),
            "percentage_distance_forming_contrails": round(
                percentage_distance_forming_contrails, 2
            ),
        },
        "number_of_flights": {
            "total": total_flights,
            "Regional": number_of_regional_flights,
            "international": number_of_international_flights,
        },
        "flight_distance_by_airspace": {
            "total_nm": float(total_distance_flown),
            "uk_airspace_nm": float(uk_airspace_segments["distance_flown_in_segment"].sum()),
            "international_airspace_nm": float(
                international_airspace_segments["distance_flown_in_segment"].sum()
            ),
        },
        "energy_forcing": {
            "total": float(total_energy_forcing),
            "uk_airspace": total_energy_forcing_in_uk_airspace,
            "international_airspace": total_energy_forcing_in_international_airspace,
        },
        "energy_forcing_per_segment": {
            "mean": mean_energy_forcing_per_segment,
            "median": median_energy_forcing_per_segment,
            "max": max_energy_forcing_per_segment,
            "mean_contrail_forming_only": mean_ef_contrail_segments,
            "median_contrail_forming_only": median_ef_contrail_segments,
        },
        "energy_forcing_per_flight": {
            "histogram": ef_histogram,
        },
        "temporal_granularity": temporal_granularity.value,
        "distance_flown_per_temporal_histogram": distance_flown_per_temporal_histogram,
        "distance_forming_contrails_per_temporal_histogram": distance_forming_contrails_per_temporal_histogram,
        "air_traffic_density_per_temporal_histogram": air_traffic_density_per_temporal_histogram,
    }

    # --- Write Output ---
    if output_filename:
        with open("results/" + output_filename + ".json", "w") as f:  # noqa: PTH123
            json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parquet_file = "2024_01_01_sample_processed_with_interpolation_with_ef"
    output_filename = "energy_forcing_statistics_sample_2024_01_01_hourly"
    generate_energy_forcing_statistics(parquet_file, output_filename, TemporalGranularity.HOURLY)
