"""Module that handles airport data for contrail avoidance model."""

from __future__ import annotations

__all__ = [
    "airport_icao_code_to_location",
    "airport_name_from_icao_code",
    "list_of_uk_airports",
    "uk_regional_flights",
]

from typing import overload

import polars as pl


def list_of_uk_airports() -> list[str]:
    """Filter the airport data to include only UK airports.

    returns: list[str]: List of ICAO codes for UK airports.
    """
    airport_data = pl.read_parquet("airport_data/airports.parquet")
    uk_airports = airport_data.filter(pl.col("iso_country") == "GB")
    return uk_airports["icao"].to_list()


def uk_regional_flights(flight_data: pl.DataFrame) -> pl.DataFrame:
    """Return the regional flights in the UK airport data."""
    list_of_uk_airports_icao = list_of_uk_airports()
    return flight_data.filter(
        pl.col("arrival_airport_icao").is_in(list_of_uk_airports_icao)
        | pl.col("departure_airport_icao").is_in(list_of_uk_airports_icao)
    )


@overload
def airport_icao_code_to_location(airport_icao_code: str) -> tuple[float, float]: ...


@overload
def airport_icao_code_to_location(airport_icao_code: list[str]) -> list[tuple[float, float]]: ...


def airport_icao_code_to_location(
    airport_icao_code: str | list[str],
) -> tuple[float, float] | list[tuple[float, float]]:
    """Get the latitude and longitude of a given airport code or codes.

    Args:
        airport_icao_code (str | list[str]): ICAO code(s) of the airport.

    Returns:
        tuple[float, float] | list[tuple[float, float]]: (latitude, longitude) of the airport(s).
    """
    airport_data = pl.read_parquet("airport_data/airports.parquet")

    if isinstance(airport_icao_code, str):
        airport_info = airport_data.filter(pl.col("icao") == airport_icao_code).select(
            ["lat", "lon"]
        )
        if airport_info.is_empty():
            msg = f"Airport code {airport_icao_code} not found."
            raise ValueError(msg)
        # explicit casts to float so mypy knows the return types
        return (float(airport_info["lat"][0]), float(airport_info["lon"][0]))

    airport_info = airport_data.filter(pl.col("icao").is_in(airport_icao_code)).select(
        ["icao", "lat", "lon"]
    )
    if airport_info.is_empty():
        msg = f"No airports found for codes: {airport_icao_code}"
        raise ValueError(msg)
    return [(float(row["lat"]), float(row["lon"])) for row in airport_info.iter_rows(named=True)]


def airport_name_from_icao_code(airport_icao_code: str) -> str:
    """Get the name of the airport given its ICAO code.

    Args:
        airport_icao_code (str): ICAO code of the airport.

    Returns:
        str: Name of the airport.
    """
    airport_data = pl.read_parquet("airport_data/airports.parquet")
    airport_info = airport_data.filter(pl.col("icao") == airport_icao_code).select(["name"])
    if airport_info.is_empty():
        msg = f"Airport code {airport_icao_code} not found."
        raise ValueError(msg)
    # cast to str to avoid returning Any
    return str(airport_info["name"][0])
