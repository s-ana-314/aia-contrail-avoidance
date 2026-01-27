"""Calculate energy forcing for flight data using the UK ADS-B January environment."""  # noqa: INP001

from __future__ import annotations

import polars as pl

from aia_model_contrail_avoidance.core_model.airspace import (
    find_airspace_of_flight_segment,
    get_gb_airspaces,
)
from aia_model_contrail_avoidance.core_model.environment import (
    calculate_total_energy_forcing,
    create_grid_environment,
    run_flight_data_through_environment,
)
from aia_model_contrail_avoidance.core_model.flights import read_ads_b_flight_dataframe


def calculate_energy_forcing_for_flights(
    parquet_file_with_ef: str,
    flight_info_with_ef_file_name: str,
    flight_dataframe_path: str | None = None,
) -> None:
    """Calculate energy forcing for flight data using the UK ADS-B January environment.

    Args:
        parquet_file_with_ef: Path to save the flight timestamps with energy forcing as a parquet
            file.
        flight_info_with_ef_file_name: Path to save the flight information with energy forcing as a
            parquet file.
        flight_dataframe_path: Optional path to a Polars DataFrame containing flight data. If None,
        it will be read from a parquet file.
    """
    # Load the processed flight data
    if flight_dataframe_path is None:
        flight_dataframe = read_ads_b_flight_dataframe()
    else:
        flight_dataframe = pl.read_parquet(flight_dataframe_path)

    print("Loading environment data...")
    environment = create_grid_environment("cocipgrid_uk_adsb_jan_result")

    # environmental bounds for UK ADS-B January environment
    environmental_bounds = {
        "lat_min": 49.0,
        "lat_max": 62.0,
        "lon_min": -8.0,
        "lon_max": 3.0,
    }

    # Remove datapoints that are outside the environment (latitude and longitude bounds)
    flight_dataframe = flight_dataframe.filter(
        (pl.col("latitude") >= environmental_bounds["lat_min"])
        & (pl.col("latitude") <= environmental_bounds["lat_max"])
        & (pl.col("longitude") >= environmental_bounds["lon_min"])
        & (pl.col("longitude") <= environmental_bounds["lon_max"])
    )
    print("\nRunning flight data through environment...")
    flight_data_with_ef = run_flight_data_through_environment(flight_dataframe, environment)
    print(f"Processed {len(flight_data_with_ef)} data points")

    # adding airspace information to dataframe
    gb_airspaces = get_gb_airspaces()
    flight_data_with_ef = find_airspace_of_flight_segment(flight_data_with_ef, gb_airspaces)
    print("Added airspace information to flight data.")

    # Save the flight data with energy forcing to parquet
    flight_data_with_ef.write_parquet(
        "data/contrails_model_data/" + parquet_file_with_ef + ".parquet"
    )

    # Calculate total energy forcing for each unique flight
    unique_flight_ids = flight_data_with_ef["flight_id"].unique().to_list()
    total_ef_list = calculate_total_energy_forcing(unique_flight_ids, flight_data_with_ef)

    # Create a summary dataframe with flight information and total energy forcing
    energy_forcing_per_flight = pl.DataFrame(
        {"flight_id": unique_flight_ids, "total_energy_forcing": total_ef_list}
    )

    flight_info_df = pl.read_parquet("data/contrails_model_data/flight_info_database.parquet")

    # Join the two dataframes on flight_id
    joined_df = flight_info_df.join(energy_forcing_per_flight, on="flight_id", how="inner")

    # Save the joined dataframe
    joined_df.write_parquet(
        "data/contrails_model_data/" + flight_info_with_ef_file_name + "_with_flight_info.parquet"
    )


if __name__ == "__main__":
    parquet_file_with_ef = "2024_01_01_sample_processed_with_interpolation_with_ef"
    output_file_name = "2024_01_01_sample_processed_with_interpolation_energy_forcing_summary"
    flight_dataframe_path = (
        "data/contrails_model_data/2024_01_01_sample_processed_with_interpolation.parquet"
    )
    calculate_energy_forcing_for_flights(
        parquet_file_with_ef,
        output_file_name,
        flight_dataframe_path=flight_dataframe_path,
    )
