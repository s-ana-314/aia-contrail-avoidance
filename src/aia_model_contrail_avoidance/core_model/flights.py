"""Generate synthetic flights."""

from __future__ import annotations

__all__ = (
    "flight_distance_from_location",
    "generate_synthetic_flight",
    "most_common_cruise_flight_level",
)

import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping


FLIGHT_DATAFRAME_SCHEMA: Mapping[str, Any] = {
    "flight_id": pl.Int64,
    "departure_location": pl.List(pl.Float64),
    "arrival_location": pl.List(pl.Float64),
    "departure_time": pl.Datetime,
    "timestamp": pl.Datetime,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "flight_level": pl.Int64,
    "distance_flown_in_segment": pl.Float64,
}


def generate_synthetic_flight(  # noqa: PLR0913
    flight_id: int,
    departure_location: tuple[float, float],
    arrival_location: tuple[float, float],
    departure_time: datetime.datetime,
    length_of_flight: float,
    flight_level: int,
) -> pl.DataFrame:
    """Generates synthetic flight from departure to arrival location as a series of timestamps.

    Args:
        flight_id: Unique identifier for the flight.
        departure_location: Tuple of (latitude, longitude) for departure.
        arrival_location: Tuple of (latitude, longitude) for arrival.
        departure_time: Departure time as a datetime object.
        length_of_flight: Length of flight in seconds.
        flight_level: Flight level as the standard pHa.
    """
    distance_traveled_in_nautical_miles = flight_distance_from_location(
        departure_location, arrival_location
    )
    number_of_timestamps = int(distance_traveled_in_nautical_miles)  # 1 nautical mile per timestamp
    latitudes = np.linspace(departure_location[0], arrival_location[0], number_of_timestamps)
    longitudes = np.linspace(departure_location[1], arrival_location[1], number_of_timestamps)
    timestamps = [
        departure_time + datetime.timedelta(seconds=i * (length_of_flight / number_of_timestamps))
        for i in range(number_of_timestamps)
    ]

    return pl.DataFrame(
        {
            "flight_id": np.full(number_of_timestamps, flight_id, dtype=int),
            "departure_location": [list(departure_location)] * number_of_timestamps,
            "arrival_location": [list(arrival_location)] * number_of_timestamps,
            "departure_time": [departure_time] * number_of_timestamps,
            "timestamp": timestamps,
            "latitude": latitudes,
            "longitude": longitudes,
            "flight_level": np.full(number_of_timestamps, flight_level, dtype=int),
            "distance_flown_in_segment": np.full(number_of_timestamps, 1.0, dtype=float),
        },
        schema=FLIGHT_DATAFRAME_SCHEMA,
    )


def flight_distance_from_location(
    departure_location: tuple[float, float] | np.ndarray,
    arrival_location: tuple[float, float] | np.ndarray,
) -> float | np.ndarray:
    """Calculates the distance between two locations using the Haversine formula.

    This is the same as the great circle distance.

    Args:
        departure_location: Tuple of (latitude, longitude) or array of shape (n, 2).
        arrival_location: Tuple of (latitude, longitude) or array of shape (n, 2).

    Returns:
        Distance in nautical miles (float or array).
    """
    earth_radius = 3443.92  # Radius of the Earth in nautical miles
    _tuple_length = 2

    departure_location = np.atleast_1d(departure_location)
    arrival_location = np.atleast_1d(arrival_location)

    # Handle scalar or tuple inputs
    if departure_location.ndim == 1 and len(departure_location) == _tuple_length:
        departure_location = departure_location.reshape(1, -1)
    if arrival_location.ndim == 1 and len(arrival_location) == _tuple_length:
        arrival_location = arrival_location.reshape(1, -1)

    departure_latitude, departure_longitude = np.radians(departure_location).T
    arrival_latitude, arrival_longitude = np.radians(arrival_location).T

    dlat = arrival_latitude - departure_latitude
    dlon = arrival_longitude - departure_longitude
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(departure_latitude) * np.cos(arrival_latitude) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    result = earth_radius * c
    return float(result[0]) if result.size == 1 else result


def most_common_cruise_flight_level() -> int:
    """Most common cruise flight level for an aircraft in UK airspace.

    Currently coinsides with sample grid data but will be updated.
    """
    return 300


def read_adsb_flight_dataframe() -> pl.DataFrame:
    """Read the pre-processed ADS-B flight data from a parquet file."""
    parquet_file = "data/contrails_model_data/2024_01_01_sample_processed.parquet"
    return pl.read_parquet(parquet_file)
