"""Calculate energy forcing for flight data using the UK ADS-B January environment."""  # noqa: INP001

from __future__ import annotations

import polars as pl

from aia_model_contrail_avoidance.core_model.environment import (
    calculate_total_energy_forcing,
    create_grid_environment_uk_adsb_jan,
    run_flight_data_through_environment,
)
from aia_model_contrail_avoidance.core_model.flights import read_adsb_flight_dataframe


def calculate_energy_forcing_for_flights(
    parquet_file_with_ef: str, flight_info_with_ef_file_name: str
) -> None:
    """Calculate energy forcing for flight data using the UK ADS-B January environment.

    Args:
        parquet_file_with_ef: Path to save the flight timestamps with energy forcing as a parquet file.
        flight_info_with_ef_file_name: Path to save the flight information with energy forcing as a parquet file.
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
    flight_dataframe_without_large_distance_segments = flight_dataframe.filter(
        pl.col("distance_flown_in_segment") <= distance_traveled_tolerance_in_meters
    )

    # Remove datapoints that are outside the UK FIR (latitude and longitude bounds)
    flight_dataframe_in_uk_airspace = flight_dataframe_without_large_distance_segments.filter(
        (pl.col("latitude") >= environmental_bounds["lat_min"])
        & (pl.col("latitude") <= environmental_bounds["lat_max"])
        & (pl.col("longitude") >= environmental_bounds["lon_min"])
        & (pl.col("longitude") <= environmental_bounds["lon_max"])
    )
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

    # Save the flight data with energy forcing to parquet
    flight_data_with_ef.write_parquet(
        "data/contrails_model_data/" + parquet_file_with_ef + ".parquet"
    )

    # Calculate total energy forcing for each unique flight
    unique_flight_ids = flight_data_with_ef["flight_id"].unique().to_list()
    total_ef_list = calculate_total_energy_forcing(unique_flight_ids, flight_data_with_ef)

    # Create a summary dataframe with flight information and total energy forcing
    ef_summary = pl.DataFrame(
        {"flight_id": unique_flight_ids, "total_energy_forcing": total_ef_list}
    )

    flight_info_df = pl.read_parquet("data/contrails_model_data/flight_info_database.parquet")

    # Join the two dataframes on flight_id
    joined_df = flight_info_df.join(ef_summary, on="flight_id", how="inner")

    # Save the joined dataframe
    joined_df.write_parquet(
        "data/contrails_model_data/" + flight_info_with_ef_file_name + "_with_flight_info.parquet"
    )


if __name__ == "__main__":
    parquet_file_with_ef = "2024_01_01_sample_with_ef"
    output_file_name = "flight_energy_forcing_summary"
    calculate_energy_forcing_for_flights(parquet_file_with_ef, output_file_name)
