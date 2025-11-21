"""Tests for generating synthetic flights."""

from __future__ import annotations

import datetime

import pytest

from aia_model_contrail_avoidance.flights import (
    flight_distance_from_location,
    generate_synthetic_flight,
    most_common_cruise_flight_level,
)


def test_flight_distance_from_location() -> None:
    """Test flight distance calculation.

    Results compared to online distance calculator
    [https://www.greatcirclemapper.net/en/great-circle-mapper/route/EGLL-EGPH/aircraft/65.html]
    """
    departure = (51.4700, -0.4543)  # London Heathrow
    arrival = (55.9533, -3.1883)  # Edinburgh
    distance = flight_distance_from_location(departure, arrival)
    assert distance == pytest.approx(288.0, rel=0.05)  # Approximate distance in nautical miles


def test_generate_synthetic_flight() -> None:
    departure_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    length_of_flight = 3600.0  # 1 hour
    expected_arrival_time = departure_time + datetime.timedelta(seconds=length_of_flight)
    heathrow_airport_location = (51.4700, -0.4543)
    edinburgh_airport_location = (55.9533, -3.1883)
    expected_distance_in_nautical_miles = 288.0

    sample_flight_dataframe = generate_synthetic_flight(
        flight_id=1,
        departure_location=heathrow_airport_location,
        arrival_location=edinburgh_airport_location,
        departure_time=departure_time,
        length_of_flight=length_of_flight,
        flight_level=most_common_cruise_flight_level(),
    )
    assert sample_flight_dataframe["flight_id"][0] == 1
    assert len(sample_flight_dataframe["timestamp"]) == pytest.approx(
        expected_distance_in_nautical_miles, rel=0.05
    )
    assert sample_flight_dataframe["timestamp"].iloc[-1] == pytest.approx(
        expected_arrival_time, abs=datetime.timedelta(seconds=60)
    )
    assert sample_flight_dataframe["latitude"][0] == pytest.approx(
        heathrow_airport_location[0], abs=1e-4
    )
    assert sample_flight_dataframe["longitude"].iloc[-1] == pytest.approx(
        edinburgh_airport_location[1], abs=1e-4
    )
    assert sample_flight_dataframe["flight_level"][0] == most_common_cruise_flight_level()
