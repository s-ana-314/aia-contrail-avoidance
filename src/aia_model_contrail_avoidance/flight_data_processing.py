"""Processing flight data from ADS-B sources into dataframes suitable for contrail modeling."""

from __future__ import annotations

import datetime
import enum
import math

import numpy as np
import polars as pl

from aia_model_contrail_avoidance.config import ADS_B_SCHEMA_CLEANED
from aia_model_contrail_avoidance.core_model.airports import list_of_uk_airports
from aia_model_contrail_avoidance.core_model.flights import flight_distance_from_location

# Global constants and enums for flight data processing
MAX_DISTANCE_BETWEEN_FLIGHT_TIMESTAMPS = 3.0  # nautical miles
LOW_FLIGHT_LEVEL_THRESHOLD = 20.0  # flight level 20 = 2000 feet


class FlightDepartureAndArrivalSubset(enum.Enum):
    """Enum for selecting subsets of flight data based on departure and arrival airports."""

    ALL = "all"
    UK = "flights to and from the UK"
    REGIONAL = "regional"


class TemporalFlightSubset(enum.Enum):
    """Enum for selecting subsets of flight data based on time periods."""

    ALL = "all"
    FIRST_MONTH = "first_month"


def process_ads_b_flight_data(
    parquet_file_path: str,
    save_filename: str,
    departure_and_arrival_subset: FlightDepartureAndArrivalSubset,
    temporal_subset: TemporalFlightSubset,
) -> None:
    """Processes ADS-B flight data from a parquet file and saves the cleaned DataFrame.

    Args:
        parquet_file_path: Path to the parquet file containing ADS-B flight data.
        save_filename: Filename (without extension) to save the processed DataFrame.
        departure_and_arrival_subset: Enum specifying the departure and arrival airport subset.
        temporal_subset: Enum specifying the temporal subset of the data.
    """
    dataframe = generate_flight_dataframe_from_ads_b_data(parquet_file_path)

    selected_dataframe = select_subset_of_ads_b_flight_data(
        dataframe, departure_and_arrival_subset, temporal_subset
    )
    cleaned_dataframe = clean_ads_b_flight_dataframe(selected_dataframe)

    process_ads_b_flight_data_for_environment(cleaned_dataframe, save_filename)

    generate_flight_info_database(save_filename, "flight_info_database")


def generate_flight_dataframe_from_ads_b_data(parquet_file_path: str) -> pl.DataFrame:
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
    print(f"INFO: Loaded flight dataframe with {len(flight_dataframe)} rows.")

    return flight_dataframe.select(needed_columns)


def select_subset_of_ads_b_flight_data(
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

    print(f"INFO: After selecting subsets, the flight dataframe has {len(flight_dataframe)} rows.")
    return flight_dataframe


def clean_ads_b_flight_dataframe(flight_dataframe: pl.DataFrame) -> pl.DataFrame:
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

    flight_dataframe = flight_dataframe.with_columns(
        pl.Series("distance_flown_in_segment", distances)
    ).drop(["prev_lat", "prev_lon"])
    # remove columns where distance_flown_in_segment is zero
    flight_dataframe = flight_dataframe.filter(pl.col("distance_flown_in_segment") > 0)

    # fill altitude nulls with previous value for that flight
    flight_dataframe = flight_dataframe.with_columns(
        pl.col("flight_level").fill_null(strategy="forward").over("flight_id")
    )
    # reorganise columns
    flight_dataframe = flight_dataframe.select(
        [
            "timestamp",
            "latitude",
            "longitude",
            "flight_level",
            "flight_id",
            "icao_address",
            "departure_airport_icao",
            "arrival_airport_icao",
            "distance_flown_in_segment",
        ]
    )
    # for large distance_flown_in_segment, create new rows with interpolated values
    flight_dataframe = generate_interpolated_rows_of_large_distance_flights(
        flight_dataframe, max_distance=MAX_DISTANCE_BETWEEN_FLIGHT_TIMESTAMPS
    )
    length_after_cleaning = len(flight_dataframe)
    print(f"INFO: After cleaning, the flight dataframe has {length_after_cleaning} rows.")

    # Merge datapoints that are very close together in space (the sum of their distances to previous and next points is less than the threshold)

    flight_dataframe = merge_close_datapoints_of_flight(
        flight_dataframe, MAX_DISTANCE_BETWEEN_FLIGHT_TIMESTAMPS
    )

    length_after_merging = len(flight_dataframe)
    print(
        f"INFO: After merging very close points, the flight dataframe has {length_after_merging} rows."
    )
    print(
        f"INFO: Total of {length_after_cleaning - length_after_merging} rows removed by merging very close points."
    )

    return flight_dataframe


def generate_interpolated_rows_of_large_distance_flights(
    flight_dataframe: pl.DataFrame, max_distance: float = 15.0
) -> pl.DataFrame:
    """Generates interpolated rows for flights with large distance flown in segment.

    Args:
        flight_dataframe: DataFrame containing ADS-B flight data.
        max_distance: Maximum distance in nautical miles before interpolation is needed.

    Returns:
        DataFrame with interpolated rows added.
    """
    previous_row = None
    for row in flight_dataframe.iter_rows(named=True):
        if row["distance_flown_in_segment"] > max_distance and previous_row is not None:
            # calculate intervals for each row
            num_new_rows = math.ceil(row["distance_flown_in_segment"] / max_distance)
            time_step = (row["timestamp"] - previous_row["timestamp"]).total_seconds() / (
                num_new_rows + 1
            )
            distance_flown_in_segment_step = row["distance_flown_in_segment"] / (num_new_rows + 1)
            # create new rows
            latitudes = np.linspace((previous_row["latitude"]), (row["latitude"]), num_new_rows)
            longitudes = np.linspace((previous_row["longitude"]), (row["longitude"]), num_new_rows)
            timestamps = [
                previous_row["timestamp"] + datetime.timedelta(seconds=i * time_step)
                for i in range(num_new_rows)
            ]
            rows_to_add = pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "latitude": latitudes,
                    "longitude": longitudes,
                    "flight_level": [previous_row["flight_level"]] * num_new_rows,
                    "flight_id": [row["flight_id"]] * num_new_rows,
                    "icao_address": [row["icao_address"]] * num_new_rows,
                    "departure_airport_icao": [row["departure_airport_icao"]] * num_new_rows,
                    "arrival_airport_icao": [row["arrival_airport_icao"]] * num_new_rows,
                    "distance_flown_in_segment": [distance_flown_in_segment_step] * num_new_rows,
                },
                schema=ADS_B_SCHEMA_CLEANED,
            )

            # add new rows to dataframe
            flight_dataframe = pl.concat([flight_dataframe, rows_to_add], how="vertical")
        previous_row = row
    # remove datapoints where distance_flown_in_segment exceeds max_distance
    flight_dataframe = flight_dataframe.filter(pl.col("distance_flown_in_segment") <= max_distance)
    # sort by flight_id and timestamp
    return flight_dataframe.sort(["flight_id", "timestamp"])


def process_ads_b_flight_data_for_environment(
    generated_dataframe: pl.DataFrame, save_filename: str
) -> None:
    """Process ADS-B flight data and save cleaned DataFrame to parquet.

    Removes datapoints with low flight levels (near or on ground)

    Args:
        generated_dataframe: DataFrame containing raw ADS-B flight data.
        save_filename: Filename (without extension) to save the processed DataFrame.
    """
    # Remove datapoints where flight level is none or negative
    dataframe_processed = generated_dataframe.filter(
        pl.col("flight_level").is_not_null()
        & (pl.col("flight_level") >= LOW_FLIGHT_LEVEL_THRESHOLD)
    )

    # percentage of datapoints removed
    percentage_removed = 100 * (1 - len(dataframe_processed) / len(generated_dataframe))
    print(f"INFO: Removed {percentage_removed:.2f}% of datapoints due to low flight level")
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


def merge_close_datapoints_of_flight(
    flight_dataframe: pl.DataFrame,
    distance_threshold: float,
) -> pl.DataFrame:
    """Merges close datapoints of a flight based on distance threshold.

    Args:
        flight_dataframe: DataFrame containing ADS-B flight data.
        distance_threshold: Distance threshold in nautical miles for merging datapoints.

    Returns:
        DataFrame with merged datapoints.
    """
    rows = list(flight_dataframe.iter_rows(named=True))
    min_n_rows = 2
    if len(rows) < min_n_rows:
        return flight_dataframe

    merged_rows = [rows[0]]
    for i in range(1, len(rows)):
        previous_row = merged_rows[-1]
        current_row = rows[i]
        # Only merge if flight_id matches and sum does not exceed threshold
        if (
            previous_row["flight_id"] == current_row["flight_id"]
            and previous_row["distance_flown_in_segment"] + current_row["distance_flown_in_segment"]
            <= distance_threshold
        ):
            # Merge: add current segment to previous
            current_row["distance_flown_in_segment"] += previous_row["distance_flown_in_segment"]
            merged_rows[-1] = current_row
        else:
            merged_rows.append(current_row)
    return pl.DataFrame(merged_rows, schema=flight_dataframe.schema)
