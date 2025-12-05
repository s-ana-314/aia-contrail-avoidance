"""Processing flight data from ADS-B sources into dataframes suitable for contrail modeling."""  # noqa: INP001

from __future__ import annotations

import pandas as pd

from aia_model_contrail_avoidance.flights import flight_distance_from_location


def generate_flight_dataframe_from_adsb_data() -> pd.DataFrame:
    """Reads ADS-B flight data into a DataFrame and creates new columns required for analysis.

    New columns added:
    - flight_level: Altitude in flight levels (altitude_baro divided by 100)
    - distance_flown_in_segment: Distance traveled in meters between consecutive datapoints for the same flight

    Returns:
        DataFrame containing ADS-B flight data.
    """
    parquet_file = "../flight-data/2024_01_01_sample.parquet"
    flight_dataframe = pd.read_parquet(parquet_file)
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
    flight_dataframe = flight_dataframe[needed_columns]

    # Divide altitude_baro by 100 to convert from pha to flight level
    flight_dataframe["altitude_baro"] = flight_dataframe["altitude_baro"] // 100.0

    # Rename altitude_baro to flight_level
    flight_dataframe = flight_dataframe.rename(columns={"altitude_baro": "flight_level"})

    # order by flight id and append distance_flown_in_segment for each datapoint by comparing it to previous with the same flightid
    flight_dataframe = flight_dataframe.sort_values(by=["flight_id", "timestamp"])

    def calculate_segment_distance(group: pd.DataFrame) -> pd.Series:
        prev_lat = group["latitude"].shift()
        prev_lon = group["longitude"].shift()

        distances: list[float] = []
        for idx, (_, row) in enumerate(group.iterrows()):
            if idx == 0 or pd.isna(prev_lat.iloc[idx]):
                distances.append(0.0)
            else:
                distance = flight_distance_from_location(
                    (row["latitude"], row["longitude"]), (prev_lat.iloc[idx], prev_lon.iloc[idx])
                )
                distances.append(float(distance))
        return pd.Series(distances, index=group.index)

    flight_dataframe["distance_flown_in_segment"] = flight_dataframe.groupby(
        "flight_id",
        group_keys=False,
    ).apply(calculate_segment_distance)

    return flight_dataframe


def process_adsb_flight_data(generated_dataframe: pd.DataFrame, save_filename: str) -> None:
    """Process ADS-B flight data and save cleaned DataFrame to parquet.

    Removes datapoints with low flight levels (near or on ground) or zero distance flown between time intervals.

    Args:
        generated_dataframe: DataFrame containing raw ADS-B flight data.
        save_filename: Filename (without extension) to save the processed DataFrame.
    """
    # Remove datapoints where flight level is none or negative
    dataframe_processed = generated_dataframe[
        generated_dataframe["flight_level"].notna() & (generated_dataframe["flight_level"] >= 0)
    ]

    # Remove datapoints where distance flown in segment is zero
    dataframe_processed = dataframe_processed[dataframe_processed["distance_flown_in_segment"] > 0]

    # percentage of datapoints removed
    percentage_removed = 100 * (1 - len(dataframe_processed) / len(generated_dataframe))
    print(
        f"INFO: Removed {percentage_removed:.2f}% of datapoints due to low flight level or zero distance flown"
    )
    # Save processed dataframe to parquet
    dataframe_processed.to_parquet("data/" + save_filename + ".parquet", index=False)


if __name__ == "__main__":
    dataframe = generate_flight_dataframe_from_adsb_data()
    save_filename = "2024_01_01_sample_processed"

    process_adsb_flight_data(dataframe, save_filename)
