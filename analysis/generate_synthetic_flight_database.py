"""Docstring for analysis.geenerate_synthetic_flight_database."""  # noqa: INP001

from __future__ import annotations

import datetime
from typing import Any

import polars as pl

from aia_model_contrail_avoidance.core_model.airports import airport_icao_code_to_location
from aia_model_contrail_avoidance.core_model.flights import generate_synthetic_flight


def generate_synthetic_flight_database(
    flight_info_list: list[dict[str, Any]], database_name: str
) -> None:
    """Generate a synthetic flight database for testing purposes."""
    flight_dataframe = pl.DataFrame()

    for flight_info in flight_info_list:
        new_flight = generate_synthetic_flight(
            flight_info["flight_id"],
            airport_icao_code_to_location(flight_info["departure_airport"]),
            airport_icao_code_to_location(flight_info["arrival_airport"]),
            flight_info["departure_time"],
            flight_info["length_of_flight"],
            flight_info["flight_level"],
        )

        flight_dataframe = pl.concat([flight_dataframe, new_flight], how="vertical")

    flight_dataframe.write_parquet(f"data/contrails_model_data/{database_name}.parquet")


if __name__ == "__main__":
    FLIGHT_INFO_LIST = [
        {
            "flight_id": 1,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 2,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 1, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 3,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 2, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 4,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 3, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 5,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 4, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 6,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 5, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 7,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 6, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 8,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 7, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 9,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 8, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 10,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 9, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 11,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 12,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 11, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 13,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 14,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 13, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 15,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 14, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 16,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 15, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 17,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 16, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 18,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 17, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 19,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 18, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 20,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 19, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 21,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 20, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 22,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 21, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 23,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 22, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
        {
            "flight_id": 24,
            "departure_airport": "EGLL",
            "arrival_airport": "EGPH",
            "departure_time": datetime.datetime(2024, 1, 1, 23, 0, 0, tzinfo=datetime.UTC),
            "length_of_flight": 1.0,
            "flight_level": 350,
        },
    ]

    generate_synthetic_flight_database(FLIGHT_INFO_LIST, "test_flights_database")
