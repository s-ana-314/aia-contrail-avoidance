"""Processing flight data from ADS-B sources into dataframes suitable for contrail modeling."""  # noqa: INP001

from __future__ import annotations

import polars as pl

from aia_model_contrail_avoidance.flights import flight_distance_from_location


def generate_flight_dataframe_from_adsb_data() -> pl.DataFrame:
    """Reads ADS-B flight data into a DataFrame and creates new columns required for analysis.

    New columns added:
    - flight_level: Altitude in flight levels (altitude_baro divided by 100)
    - distance_flown_in_segment: Distance traveled in meters between consecutive datapoints for the same flight

    Returns:
        DataFrame containing ADS-B flight data.
    """
    parquet_file = "../flight-data/2024_01_01_sample.parquet"
    flight_dataframe = pl.read_parquet(parquet_file)
    # keep needed columns only
    needed_columns = [
        "timestamp",
        "latitude",
        "longitude",
        "altitude_baro",
        "flight_id",
        "aircraft_type_icao",
        "departure_airport_icao",
        "arrival_airport_icao",
    ]
    flight_dataframe = flight_dataframe.select(needed_columns)

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
    dataframe_processed.write_parquet("data/" + save_filename + ".parquet")


if __name__ == "__main__":
    dataframe = generate_flight_dataframe_from_adsb_data()
    save_filename = "2024_01_01_sample_processed"

    process_adsb_flight_data(dataframe, save_filename)
