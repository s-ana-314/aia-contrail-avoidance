"""Module that handles airport data for contrail avoidance model."""

from __future__ import annotations

__all__ = ["list_of_uk_airports", "uk_regional_flights"]

import polars as pl


def list_of_uk_airports() -> list[str]:
    """Filter the airport data to include only UK airports."""
    airport_data = pl.read_parquet("../airport_data/airports.parquet")
    uk_airports = airport_data.filter(pl.col("iso_country") == "GB")
    return uk_airports["icao"].to_list()


def uk_regional_flights(flight_data: pl.DataFrame) -> pl.DataFrame:
    """Return the regional flights in the UK airport data."""
    list_of_uk_airports_icao = list_of_uk_airports()
    return flight_data.filter(
        pl.col("arrival_airport_icao").is_in(list_of_uk_airports_icao)
        | pl.col("departure_airport_icao").is_in(list_of_uk_airports_icao)
    )
