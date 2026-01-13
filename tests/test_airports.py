"""Tests for interfacing with airport data."""

from __future__ import annotations

import pytest

from aia_model_contrail_avoidance.airports import (
    airport_icao_code_to_location,
    airport_name_from_icao_code,
    list_of_uk_airports,
)


def test_list_of_uk_airports() -> None:
    """Test that the list_of_uk_airports function returns >40 UK airports."""
    minimum_uk_airports = 40.0
    uk_airports = list_of_uk_airports()
    assert len(uk_airports) > minimum_uk_airports
    assert all(isinstance(code, str) for code in uk_airports)


def test_airport_icao_code_to_location() -> None:
    """Test that the airport_icao_code_to_location function returns correct lat/lon."""
    tolerance = 0.01
    lat_long = airport_icao_code_to_location("EGLL")  # London Heathrow
    lat, lon = lat_long
    assert abs(lat - 51.47) < tolerance
    assert abs(lon - (-0.46)) < tolerance


@pytest.mark.parametrize(
    ("icao_code", "expected_name"),
    (
        ("EGLL", "London Heathrow Airport"),
        ("EGKK", "London Gatwick Airport"),
        ("EGSS", "London Stansted Airport"),
    ),
)
def test_airport_name_from_icao_code(icao_code: str, expected_name: str) -> None:
    """Test that the airport_name_from_icao_code function returns correct airport name."""
    name = airport_name_from_icao_code(icao_code)
    assert name == expected_name
