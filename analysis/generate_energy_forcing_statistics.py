"""Generate energy forcing summary statistics including contrail formation analysis."""  # noqa: INP001

from __future__ import annotations

import json

import numpy as np
import polars as pl


def generate_energy_forcing_statistics(
    parquet_file: str, output_filename: str | None = None
) -> None:
    """Generate energy forcing summary statistics including contrail formation analysis.

    Args:
        parquet_file: Path to the parquet file containing flight data with energy forcing.
        output_filename: Optional path to save the statistics as JSON. If None, no file is written.

    """
    # Load the flight data with energy forcing
    flight_dataframe = pl.read_parquet(f"data/{parquet_file}.parquet")

    # --- Basic Energy Forcing Statistics ---
    total_datapoints = len(flight_dataframe)
    total_flights = flight_dataframe["flight_id"].n_unique()

    # Calculate total distance flown
    total_distance_flown = flight_dataframe["distance_flown_in_segment"].sum()

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
    # distance flown per hpour of the day
    distance_flown_per_hour_histogram = None
    if (
        "timestamp" in flight_dataframe.columns
        and "distance_flown_in_segment" in flight_dataframe.columns
    ):
        flight_dataframe_with_hour = flight_dataframe.with_columns(
            pl.col("timestamp").dt.hour().alias("hour")
        )
        distance_per_hour = (
            flight_dataframe_with_hour.group_by("hour")
            .agg(pl.col("distance_flown_in_segment").sum())
            .to_dict(as_series=False)
        )
        hour_to_distance = dict(
            zip(
                distance_per_hour["hour"],
                distance_per_hour["distance_flown_in_segment"],
                strict=True,
            )
        )
        distance_flown_per_hour_histogram = {
            str(hour): hour_to_distance.get(hour, 0) for hour in range(24)
        }
    # histogram of distance forming contrails per hour of the day
    if "timestamp" in contrail_forming_segments.columns:
        contrail_forming_segments_with_hour = contrail_forming_segments.with_columns(
            pl.col("timestamp").dt.hour().alias("hour")
        )

        distance_per_hour = (
            contrail_forming_segments_with_hour.group_by("hour")
            .agg(pl.col("distance_flown_in_segment").sum())
            .to_dict(as_series=False)
        )

        hour_to_distance = dict(
            zip(
                distance_per_hour["hour"],
                distance_per_hour["distance_flown_in_segment"],
                strict=True,
            )
        )

        # Ensure all hours 0-23 are present (fill missing hours with 0)
        distance_forming_contrails_per_hour_histogram = {
            str(hour): hour_to_distance.get(hour, 0) for hour in range(24)
        }
    else:
        distance_forming_contrails_per_hour_histogram = None

    # --- Build Summary ---
    stats = {
        "file_name": parquet_file,
        "overview": {
            "total_datapoints": total_datapoints,
            "total_flights": total_flights,
            "total_distance_flown_nm": float(total_distance_flown),
            "total_energy_forcing": float(total_energy_forcing),
        },
        "contrail_formation": {
            "flights_forming_contrails": int(flights_forming_contrails),
            "percentage_flights_forming_contrails": round(percentage_flights_forming_contrails, 2),
            "distance_forming_contrails_nm": float(distance_forming_contrails),
            "percentage_distance_forming_contrails": round(
                percentage_distance_forming_contrails, 2
            ),
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
        "distance_flown_per_hour_histogram": distance_flown_per_hour_histogram,
        "distance_forming_contrails_per_hour_histogram": distance_forming_contrails_per_hour_histogram,
    }

    # --- Write Output ---
    if output_filename:
        with open("results/" + output_filename + ".json", "w") as f:  # noqa: PTH123
            json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parquet_file = "2024_01_01_sample_with_ef"
    output_filename = "energy_forcing_statistics"
    generate_energy_forcing_statistics(parquet_file, output_filename)
