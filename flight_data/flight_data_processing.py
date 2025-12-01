"""Processing flight data from ADS-B sources into dataframes suitable for contrail modeling."""  # noqa: INP001

from __future__ import annotations

import pandas as pd

from aia_model_contrail_avoidance.flights import flight_distance_from_location


def generate_flight_dataframe_from_adsb_data() -> pd.DataFrame:
    """Reads ADS-B flight data into a DataFrame and creates new columns.

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
        "flight_id", group_keys=False
    ).apply(calculate_segment_distance)

    return flight_dataframe


dataframe = generate_flight_dataframe_from_adsb_data()

# show variation in distance flown between rows
print(dataframe["distance_flown_in_segment"].describe())

# remove datapoints where distance flown in segment is zero within 1 percent tolerance
dataframe_processed = dataframe[dataframe["distance_flown_in_segment"] > 0.00]

# percentage of datapoints removed
removed_percentage = (1 - len(dataframe_processed) / len(dataframe)) * 100
print(f"Removed {removed_percentage:.2f}% of datapoints with zero distance flown in segment.")

# save processed dataframe to parquet for faster loading next time
dataframe_processed.to_parquet("2024_01_01_sample_processed.parquet")
