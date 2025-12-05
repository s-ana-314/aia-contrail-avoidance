"""Calculate energy forcing for flight data using the UK ADS-B January environment."""  # noqa: INP001

from __future__ import annotations

import pandas as pd

from aia_model_contrail_avoidance.environment import (
    calculate_total_energy_forcing,
    create_grid_environment_uk_adsb_jan,
    run_flight_data_through_environment,
)
from aia_model_contrail_avoidance.flights import read_adsb_flight_dataframe


def calculate_energy_forcing_for_flights(parquet_file_with_ef: str, output_file_name: str) -> None:
    """Calculate energy forcing for flight data using the UK ADS-B January environment.

    Args:
        parquet_file_with_ef: Path to save the flight data with energy forcing as a parquet file.
        output_file_name: Path to save the energy forcing summary statistics as a parquet file.
    """
    # Load the processed flight data
    flight_dataframe = read_adsb_flight_dataframe()

    distance_traveled_tolerance_in_meters = 2000

    environmental_bounds = {
        "lat_min": 49.0,
        "lat_max": 62.0,
        "lon_min": -8.0,
        "lon_max": 3.0,
    }

    # Remove datapoints where distance_traveled_in_segment is > tolerance (2000 m)
    flight_dataframe_without_large_distance_segments = flight_dataframe[
        flight_dataframe["distance_flown_in_segment"] <= distance_traveled_tolerance_in_meters
    ]

    # Remove datapoints that are outside the UK FIR (latitude and longitude bounds)
    flight_dataframe_in_uk_airspace = flight_dataframe_without_large_distance_segments[
        (
            flight_dataframe_without_large_distance_segments["latitude"]
            >= environmental_bounds["lat_min"]
        )
        & (
            flight_dataframe_without_large_distance_segments["latitude"]
            <= environmental_bounds["lat_max"]
        )
        & (
            flight_dataframe_without_large_distance_segments["longitude"]
            >= environmental_bounds["lon_min"]
        )
        & (
            flight_dataframe_without_large_distance_segments["longitude"]
            <= environmental_bounds["lon_max"]
        )
    ]
    # informative statistics
    percentage_removed_due_to_large_distances_flown = 100 * (
        1 - len(flight_dataframe_without_large_distance_segments) / len(flight_dataframe)
    )
    percentage_removed_due_to_uk_airspace = 100 * (
        1 - len(flight_dataframe_in_uk_airspace) / len(flight_dataframe)
    )

    print(
        f"INFO: Removed {percentage_removed_due_to_large_distances_flown:.2f}% of data points due to large distances flown in segment (> {distance_traveled_tolerance_in_meters} m)"
    )
    print(
        f"INFO: Removed {percentage_removed_due_to_uk_airspace:.2f}% of data points outside UK airspace"
    )

    print("Loading environment data...")
    environment = create_grid_environment_uk_adsb_jan()

    print("\nRunning flight data through environment...")
    flight_data_with_ef = run_flight_data_through_environment(
        flight_dataframe_in_uk_airspace, environment
    )
    print(f"Processed {len(flight_data_with_ef)} data points")

    # Calculate total energy forcing for each unique flight
    unique_flight_ids = flight_data_with_ef["flight_id"].unique().tolist()
    total_ef_list = calculate_total_energy_forcing(unique_flight_ids, flight_data_with_ef)

    # Create a summary dataframes
    ef_summary = pd.DataFrame(
        {"flight_id": unique_flight_ids, "total_energy_forcing": total_ef_list}
    )

    # Save the full dataset with energy forcing
    flight_data_with_ef.to_parquet("data/" + parquet_file_with_ef + ".parquet", index=False)

    # Save the summary dataframe
    ef_summary.to_parquet("data/" + output_file_name + ".parquet", index=False)


if __name__ == "__main__":
    parquet_file_with_ef = "2024_01_01_sample_with_ef"
    output_file_name = "flight_energy_forcing_summary"
    calculate_energy_forcing_for_flights(parquet_file_with_ef, output_file_name)
