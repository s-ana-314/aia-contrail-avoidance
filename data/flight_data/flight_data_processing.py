"""Processing flight data from ADS-B sources into dataframes suitable for contrail modeling."""  # noqa: INP001

from __future__ import annotations

import datetime
import enum
import math

import numpy as np
import polars as pl

from aia_model_contrail_avoidance.config import ADS_B_SCHEMA_CLEANED
from aia_model_contrail_avoidance.core_model.airports import list_of_uk_airports
from aia_model_contrail_avoidance.core_model.flights import flight_distance_from_location


class FlightDepartureAndArrivalSubset(enum.Enum):
    """Enum for selecting subsets of flight data based on departure and arrival airports."""

    ALL = "all"
    UK = "flights to and from the UK"
    REGIONAL = "regional"


class TemporalFlightSubset(enum.Enum):
    """Enum for selecting subsets of flight data based on time periods."""

    ALL = "all"
    FIRST_MONTH = "first_month"


def generate_flight_dataframe_from_adsb_data(parquet_file_path: str) -> pl.DataFrame:
    """Reads ADS-B flight data into a DataFrame and removes unnecessary columns.

    Args:
        parquet_file_path: Path to the parquet file containing ADS-B flight data.

    Returns:
        DataFrame containing ADS-B flight data.
    """
    flight_dataframe = pl.read_parquet(parquet_file_path)
    needed_columns = [
        "timestamp",
        "latitude",
        "longitude",
        "altitude_baro",
        "flight_id",
        "icao_address",
        "departure_airport_icao",
        "arrival_airport_icao",
    ]

    return flight_dataframe.select(needed_columns)


def select_subset_of_adsb_flight_data(
    flight_dataframe: pl.DataFrame,
    departure_and_arrival_subset: FlightDepartureAndArrivalSubset,
    temporal_subset: TemporalFlightSubset,
) -> pl.DataFrame:
    """Selects a subset of columns from the ADS-B flight data DataFrame.

    Args:
        flight_dataframe: DataFrame containing ADS-B flight data.
        departure_and_arrival_subset: Enum specifying the departure and arrival airport subset.
        temporal_subset: Enum specifying the temporal subset of the data.

    Returns:
        DataFrame containing a subset of the original ADS-B flight data.
    """
    if temporal_subset == TemporalFlightSubset.FIRST_MONTH:
        flight_dataframe = flight_dataframe.filter(
            (pl.col("timestamp") >= pl.datetime(2024, 1, 1, time_zone="UTC"))
            & (pl.col("timestamp") < pl.datetime(2024, 2, 1, time_zone="UTC"))
        )

    if departure_and_arrival_subset == FlightDepartureAndArrivalSubset.UK:
        uk_airport_icaos = list_of_uk_airports()
        flight_dataframe = flight_dataframe.filter(
            pl.col("arrival_airport_icao").is_in(uk_airport_icaos)
            | pl.col("departure_airport_icao").is_in(uk_airport_icaos)
        )

    elif departure_and_arrival_subset == FlightDepartureAndArrivalSubset.REGIONAL:
        uk_airport_icaos = list_of_uk_airports()
        flight_dataframe = flight_dataframe.filter(
            pl.col("arrival_airport_icao").is_in(uk_airport_icaos)
            & pl.col("departure_airport_icao").is_in(uk_airport_icaos)
        )

    return flight_dataframe


def clean_adsb_flight_dataframe(flight_dataframe: pl.DataFrame) -> pl.DataFrame:
    """Cleans the flight DataFrame by adding necessary columns and removing unnecessary ones.

    New columns added:
    - flight_level: Altitude of aircraft in term of flight levels (altitude_baro divided by 100)
    - distance_flown_in_segment: Distance traveled in meters between consecutive datapoints for the
        same flight

    Augmented rows:
    - For any row where distance_flown_in_segment exceeds a threshold (e.g. 50 nautical miles), new
        rows are generated with interpolated values for latitude, longitude, and timestamp to ensure
        no segment exceeds the threshold.

    Args:
        flight_dataframe: DataFrame containing ADS-B flight data.

    Returns:
        Cleaned DataFrame with added columns.
    """
    # Divide altitude_baro by 100 to convert from pha to flight level
    flight_dataframe = flight_dataframe.with_columns(
        (pl.col("altitude_baro") // 100.0).alias("flight_level")
    )

    # Drop the original altitude_baro column
    flight_dataframe = flight_dataframe.drop("altitude_baro")

    # order by flight id and timestamp
    flight_dataframe = flight_dataframe.sort(["flight_id", "timestamp"])

    # Calculate distance_flown_in_segment using window functions
    flight_dataframe = flight_dataframe.with_columns(
        [
            pl.col("latitude").shift(1).over("flight_id").alias("prev_lat"),
            pl.col("longitude").shift(1).over("flight_id").alias("prev_lon"),
        ]
    )

    # Calculate distances for each row
    distances = []
    for row in flight_dataframe.iter_rows(named=True):
        if row["prev_lat"] is None or row["prev_lon"] is None:
            distances.append(0.0)
        else:
            distance = flight_distance_from_location(
                (row["latitude"], row["longitude"]), (row["prev_lat"], row["prev_lon"])
            )
            distances.append(float(distance))

    return flight_dataframe.with_columns(pl.Series("distance_flown_in_segment", distances)).drop(
        ["prev_lat", "prev_lon"]
    )


def process_adsb_flight_data(generated_dataframe: pl.DataFrame, save_filename: str) -> None:
    """Process ADS-B flight data and save cleaned DataFrame to parquet.

    Removes datapoints with low flight levels (near or on ground) or zero distance flown between time intervals.

    Args:
        generated_dataframe: DataFrame containing raw ADS-B flight data.
        save_filename: Filename (without extension) to save the processed DataFrame.
    """
    # Remove datapoints where flight level is none or negative
    dataframe_processed = generated_dataframe.filter(
        pl.col("flight_level").is_not_null() & (pl.col("flight_level") >= 0)
    )

    # Remove datapoints where distance flown in segment is zero
    dataframe_processed = dataframe_processed.filter(pl.col("distance_flown_in_segment") > 0)

    # percentage of datapoints removed
    percentage_removed = 100 * (1 - len(dataframe_processed) / len(generated_dataframe))
    print(
        f"INFO: Removed {percentage_removed:.2f}% of datapoints due to low flight level or zero distance flown"
    )
    # Save processed dataframe to parquet
    dataframe_processed.write_parquet("data/contrails_model_data/" + save_filename + ".parquet")


def generate_flight_info_database(processed_parquet_filename: str, save_filename: str) -> None:
    """Generates a flight information database from processed ADS-B data."""
    processed_parquet_file = "data/contrails_model_data/" + processed_parquet_filename + ".parquet"
    flight_dataframe = pl.read_parquet(processed_parquet_file)

    # Extract unique flight information
    flight_info_df = flight_dataframe.select(
        [
            "flight_id",
            "icao_address",
            "departure_airport_icao",
            "arrival_airport_icao",
        ]
    ).unique()

    # add new colums for first message timestamp and last message timestamp
    first_timestamps = flight_dataframe.group_by("flight_id").agg(
        pl.col("timestamp").min().alias("first_message_timestamp")
    )
    last_timestamps = flight_dataframe.group_by("flight_id").agg(
        pl.col("timestamp").max().alias("last_message_timestamp")
    )
    flight_info_df = flight_info_df.join(first_timestamps, on="flight_id").join(
        last_timestamps, on="flight_id"
    )

    # Save flight information database to parquet
    flight_info_df.write_parquet("data/contrails_model_data/" + save_filename + ".parquet")


if __name__ == "__main__":
    dataframe = generate_flight_dataframe_from_adsb_data()
    save_filename = "2024_01_01_sample_processed"

    process_adsb_flight_data(dataframe, save_filename)

    generate_flight_info_database(save_filename, "flight_info_database")
