"""Generate energy forcing summary statistics including contrail formation analysis."""  # noqa: INP001

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
)


def create_histogram_distance_flown_over_time(
    temporal_granularity: TemporalGranularity, flight_dataframe: pl.DataFrame
) -> dict[str, float]:
    """Calculate distance flown per temporal unit histogram.

    Args:
        temporal_granularity: Temporal granularity for aggregation.
        flight_dataframe: DataFrame containing flight data.

    Returns:
        Dictionary mapping temporal units to distance flown.
    """
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
    return {str(unit): temporal_to_distance.get(unit, 0) for unit in temporal_range}


def create_histogram_distance_forming_contrails_over_time(
    temporal_granularity: TemporalGranularity, flight_dataframe_of_contrails: pl.DataFrame
) -> dict[str, float]:
    """Calculate distance forming contrails per temporal unit histogram.

    Args:
        temporal_granularity: Temporal granularity for aggregation.
        flight_dataframe_of_contrails: DataFrame containing flight data forming contrails.

    Returns:
        Dictionary mapping temporal units to distance forming contrails.
    """
    temporal_field = _get_temporal_grouping_field(temporal_granularity)
    temporal_range, _labels = _get_temporal_range_and_labels(temporal_granularity)

    flight_dataframe_with_temporal = flight_dataframe_of_contrails.with_columns(
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
    return {str(unit): temporal_to_distance.get(unit, 0) for unit in temporal_range}


def create_histogram_cumulative_energy_forcing_flight(
    flight_dataframe: pl.DataFrame,
) -> dict[str, float]:
    """Calculate cumulative energy forcing per flight histogram.

    Args:
        flight_dataframe: DataFrame containing flight data.

    Returns:
        Dictionary mapping temporal units to cumulative flight distance.
    """
    # Per-flight energy forcing statistics
    flight_ef_summary = flight_dataframe.group_by("flight_id").agg(
        pl.col("ef").sum().alias("total_ef")
    )
    ef_values = flight_ef_summary["total_ef"].to_numpy().astype("float64")
    hist_counts, bin_edges = np.histogram(ef_values, bins=50, density=False)
    cumulative_counts = hist_counts.cumsum()
    return {
        "bin_edges": bin_edges.tolist(),
        "counts": hist_counts.tolist(),
        "cumulative_counts": cumulative_counts.tolist(),
    }


def create_histogram_air_traffic_density_over_time(
    temporal_granularity: TemporalGranularity, flight_dataframe: pl.DataFrame
) -> dict[str, int]:
    """Calculate air traffic density per temporal unit histogram.

    Args:
        temporal_granularity: Temporal granularity for aggregation.
        flight_dataframe: DataFrame containing flight data.

    Returns:
        Dictionary mapping temporal units to air traffic density.
    """
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
    return {str(unit): temporal_to_planes.get(unit, 0) for unit in temporal_range}


def generate_energy_forcing_statistics(
    complete_flight_dataframe: pl.DataFrame,
    output_filename: str | None = None,
) -> None:
    """Generate energy forcing summary statistics including contrail formation analysis.

    Args:
        complete_flight_dataframe: DataFrame containing flight data with energy forcing.
        output_filename: Optional path to save the statistics as JSON. If None, no file is written.

    """
    # --- General Statistics ---
    total_number_of_datapoints = len(complete_flight_dataframe)
    total_number_of_flights = complete_flight_dataframe["flight_id"].n_unique()
    total_distance_flown = complete_flight_dataframe["distance_flown_in_segment"].sum()

    # -- Airspace Specific Statistics ---
    uk_airspace_flights_dataframe = complete_flight_dataframe.filter(
        pl.col("airspace").is_not_null()
    )

    international_airspace_flights_dataframe = complete_flight_dataframe.filter(
        pl.col("airspace").is_null()
    )

    total_energy_forcing_in_uk_airspace = uk_airspace_flights_dataframe["ef"].sum()
    total_energy_forcing_in_international_airspace = international_airspace_flights_dataframe[
        "ef"
    ].sum()

    # -- Regional vs International Flights ---
    uk_airports = list_of_uk_airports()
    regional_flights_dataframe = complete_flight_dataframe.filter(
        pl.col("arrival_airport_icao").is_in(uk_airports)
        & pl.col("departure_airport_icao").is_in(uk_airports)
    )

    number_of_regional_flights = regional_flights_dataframe["flight_id"].n_unique()
    number_of_international_flights = total_number_of_flights - number_of_regional_flights

    # --- Contrail Formation Analysis ---

    contrail_forming_flight_segments_dataframe = complete_flight_dataframe.filter(
        pl.col("ef") > 0
    )  # Segments with positive energy forcing form contrails

    total_distance_forming_contrails = contrail_forming_flight_segments_dataframe[
        "distance_flown_in_segment"
    ].sum()
    percentage_distance_forming_contrails = (
        (total_distance_forming_contrails / total_distance_flown) * 100
        if total_distance_flown > 0
        else 0
    )

    number_of_flights_forming_contrails = contrail_forming_flight_segments_dataframe[
        "flight_id"
    ].n_unique()
    percentage_of_flights_forming_contrails = (
        number_of_flights_forming_contrails / total_number_of_flights
    ) * 100

    # --- Energy Forcing Statistics ---
    total_energy_forcing = contrail_forming_flight_segments_dataframe["ef"].sum()

    # --- Build Summary ---
    stats = {
        "overview": {
            "total_datapoints": total_number_of_datapoints,
        },
        "contrail_formation": {
            "flights_forming_contrails": int(number_of_flights_forming_contrails),
            "percentage_flights_forming_contrails": round(
                percentage_of_flights_forming_contrails, 2
            ),
            "distance_forming_contrails_nm": float(total_distance_forming_contrails),
            "percentage_distance_forming_contrails": round(
                percentage_distance_forming_contrails, 2
            ),
        },
        "number_of_flights": {
            "total": total_number_of_flights,
            "regional": number_of_regional_flights,
            "international": number_of_international_flights,
        },
        "flight_distance_by_airspace": {
            "total_nm": float(total_distance_flown),
            "uk_airspace_nm": float(
                uk_airspace_flights_dataframe["distance_flown_in_segment"].sum()
            ),
            "international_airspace_nm": float(
                international_airspace_flights_dataframe["distance_flown_in_segment"].sum()
            ),
        },
        "energy_forcing": {
            "total": float(total_energy_forcing),
            "uk_airspace": float(total_energy_forcing_in_uk_airspace),
            "international_airspace": float(total_energy_forcing_in_international_airspace),
        },
        "Cumulative_energy_forcing_per_flight": {
            "histogram": create_histogram_cumulative_energy_forcing_flight(
                complete_flight_dataframe
            ),
        },
        "distance_flown_over_time_histogram": {
            "hourly": create_histogram_distance_flown_over_time(
                TemporalGranularity.HOURLY, complete_flight_dataframe
            ),
            "daily": create_histogram_distance_flown_over_time(
                TemporalGranularity.DAILY, complete_flight_dataframe
            ),
            "monthly": create_histogram_distance_flown_over_time(
                TemporalGranularity.MONTHLY, complete_flight_dataframe
            ),
            "seasonally": create_histogram_distance_flown_over_time(
                TemporalGranularity.SEASONALLY, complete_flight_dataframe
            ),
            "annually": create_histogram_distance_flown_over_time(
                TemporalGranularity.ANNUALLY, complete_flight_dataframe
            ),
        },
        "distance_forming_contrails_over_time_histogram": {
            "hourly": create_histogram_distance_forming_contrails_over_time(
                TemporalGranularity.HOURLY, contrail_forming_flight_segments_dataframe
            ),
            "daily": create_histogram_distance_forming_contrails_over_time(
                TemporalGranularity.DAILY, contrail_forming_flight_segments_dataframe
            ),
            "monthly": create_histogram_distance_forming_contrails_over_time(
                TemporalGranularity.MONTHLY, contrail_forming_flight_segments_dataframe
            ),
            "seasonally": create_histogram_distance_forming_contrails_over_time(
                TemporalGranularity.SEASONALLY, contrail_forming_flight_segments_dataframe
            ),
            "annually": create_histogram_distance_forming_contrails_over_time(
                TemporalGranularity.ANNUALLY, contrail_forming_flight_segments_dataframe
            ),
        },
        "air_traffic_density_over_time_histogram": {
            "hourly": create_histogram_air_traffic_density_over_time(
                TemporalGranularity.HOURLY, complete_flight_dataframe
            ),
            "daily": create_histogram_air_traffic_density_over_time(
                TemporalGranularity.DAILY, complete_flight_dataframe
            ),
            "monthly": create_histogram_air_traffic_density_over_time(
                TemporalGranularity.MONTHLY, complete_flight_dataframe
            ),
            "seasonally": create_histogram_air_traffic_density_over_time(
                TemporalGranularity.SEASONALLY, complete_flight_dataframe
            ),
            "annually": create_histogram_air_traffic_density_over_time(
                TemporalGranularity.ANNUALLY, complete_flight_dataframe
            ),
        },
    }

    # --- Write Output ---
    if output_filename:
        with Path("results/" + output_filename + ".json").open("w") as f:
            json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parquet_file = "2024_01_01_sample_processed_with_interpolation_with_ef"
    output_filename = "energy_forcing_statistics_sample_2024_01_01"

    complete_flight_dataframe = pl.read_parquet(f"data/contrails_model_data/{parquet_file}.parquet")

    generate_energy_forcing_statistics(complete_flight_dataframe, output_filename)
