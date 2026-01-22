"""Configuration for the flight data schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

    from polars.type_aliases import PolarsDataType


ADS_B_PARQUET_SCHEMA_WITH_FLIGHT_ID: dict[str, PolarsDataType] = {
    "timestamp": pl.Datetime,
    "source": pl.String,
    "flight_id": pl.Int32,
    "callsign": pl.String,
    "icao_address": pl.String,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "altitude_baro": pl.Int32,
    "altitude_gnss": pl.Int32,
    "on_ground": pl.Boolean,
    "heading": pl.Float32,
    "speed": pl.Int32,
    "vertical_rate": pl.Int32,
    "squawk": pl.String,
    "aircraft_type_icao": pl.String,
    "aircraft_type_name": pl.String,
    "airline_iata": pl.String,
    "flight_number": pl.String,
    "departure_airport_icao": pl.String,
    "arrival_airport_icao": pl.String,
    "departure_scheduled_time": pl.Datetime,
    "arrival_scheduled_time": pl.Datetime,
    "takeoff_time": pl.Datetime,
    "landing_time": pl.Datetime,
}

ADS_B_SCHEMA_CLEANED: dict[str, PolarsDataType] = {
    "timestamp": pl.Datetime,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "flight_level": pl.Float64,
    "flight_id": pl.UInt32,
    "icao_address": pl.String,
    "departure_airport_icao": pl.String,
    "arrival_airport_icao": pl.String,
    "distance_flown_in_segment": pl.Float64,
}

FLIGHT_INFORMATION_SCHEMA: dict[str, PolarsDataType] = {
    "flight_id": pl.Int32,
    "aircraft_type_icao": pl.String,
    "aircraft_type_name": pl.String,
    "airline_iata": pl.String,
    "flight_number": pl.String,
    "departure_airport_icao": pl.String,
    "arrival_airport_icao": pl.String,
    "departure_scheduled_time": pl.Datetime,
    "arrival_scheduled_time": pl.Datetime,
    "takeoff_time": pl.Datetime,
    "landing_time": pl.Datetime,
}

FLIGHT_TIMESTAMPS_SCHEMA: Mapping[str, PolarsDataType] = {
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
