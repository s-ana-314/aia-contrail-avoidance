"""Module that handles airport data for contrail avoidance model."""

from __future__ import annotations

__all__ = ["list_of_uk_airports", "uk_regional_flights"]

import pandas as pd


def list_of_uk_airports() -> list[str]:
    """Filter the airport data to include only UK airports."""
    airport_data = pd.read_parquet("airport_data/airports.parquet")
    uk_airports = airport_data[airport_data["iso_country"] == "GB"]
    return uk_airports["icao"].tolist()


def uk_regional_flights(flight_data: pd.DataFrame) -> pd.DataFrame:
    """Return the regional flights in the UK airport data."""
    list_of_uk_airports_icao = list_of_uk_airports()
    return flight_data[
        flight_data["arrival_airport_icao"].isin(list_of_uk_airports_icao)
        | flight_data["departure_airport_icao"].isin(list_of_uk_airports_icao)
    ]
