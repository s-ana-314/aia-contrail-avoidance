"""Tests for generating environment and erf data."""

from __future__ import annotations

import datetime

import pytest

from aia_model_contrail_avoidance.environment import (
    calculate_effective_radiative_forcing,
    create_grid_environment,
    create_synthetic_grid_environment,
    run_flight_data_through_environment,
)
from aia_model_contrail_avoidance.flights import (
    generate_synthetic_flight,
    most_common_cruise_flight_level,
)


def test_create_grid_environment() -> None:
    """Test creating grid environment."""
    environment = create_grid_environment()
    assert environment.name == "ef_per_m"
    assert all(dim in environment.dims for dim in ("time", "level", "latitude", "longitude"))


def test_create_synthetic_grid_environment() -> None:
    """Test creating grid environment."""
    expected_low_ef = 0.0
    expected_mid_ef = 0.5
    test_high_ef = 1.0
    reference_time = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=None)  # noqa: DTZ001
    reference_latitude = 50.0
    reference_longitude = -1.0

    environment = create_synthetic_grid_environment()
    assert (
        environment.sel(
            level=250,
            time=reference_time,
            latitude=reference_latitude,
            longitude=reference_longitude,
        ).item()
        == expected_low_ef
    )
    assert (
        environment.sel(
            level=300,
            time=reference_time,
            latitude=reference_latitude,
            longitude=reference_longitude,
        ).item()
        == expected_mid_ef
    )
    assert (
        environment.sel(
            level=350,
            time=reference_time,
            latitude=reference_latitude,
            longitude=reference_longitude,
        ).item()
        == test_high_ef
    )


def test_run_flight_data_through_environment() -> None:
    """Test running flight data through environment."""
    environment = create_synthetic_grid_environment()
    departure_time = datetime.datetime(2024, 1, 1, 1, 0, 0, tzinfo=datetime.UTC)
    length_of_flight = 3600.0  # 1 hour
    departure_location = (48.0, -110.0)
    destination_location = (48.0, -102.5)

    sample_flight_dataframe = generate_synthetic_flight(
        flight_id=1,
        departure_location=departure_location,
        arrival_location=destination_location,
        departure_time=departure_time,
        length_of_flight=length_of_flight,
        flight_level=most_common_cruise_flight_level(),
    )

    flight_with_erf = run_flight_data_through_environment(sample_flight_dataframe, environment)
    assert "erf" in flight_with_erf.columns
    assert not flight_with_erf["erf"].isnull().any()


@pytest.mark.parametrize(
    ("flight_level", "expected_total_ef"),
    (
        (250, 0.0),
        (300, 288.0 / 2.0),
        (350, 288.0),
    ),
)
def test_calculate_effective_radiative_forcing(flight_level: int, expected_total_ef: float) -> None:
    """Validate the total effective radiative forcing from sample synthetic flights.

    Note: the Ef is wrong by a conversion factor from meters to nautical miles.
    The main aim of this test is to determine if the functions are correctly sampling the
        environment.
    """
    environment = create_synthetic_grid_environment()
    departure_time = datetime.datetime(2024, 1, 1, 1, 0, 0, tzinfo=datetime.UTC)
    length_of_flight = 3600.0  # 1 hour
    heathrow_airport_location = (51.4700, -0.4543)
    edinburgh_airport_location = (55.9533, -3.1883)

    sample_flight_dataframe = generate_synthetic_flight(
        flight_id=1,
        departure_location=heathrow_airport_location,
        arrival_location=edinburgh_airport_location,
        departure_time=departure_time,
        length_of_flight=length_of_flight,
        flight_level=flight_level,
    )

    flight_with_erf = run_flight_data_through_environment(sample_flight_dataframe, environment)

    total_ef = calculate_effective_radiative_forcing(1, flight_with_erf)
    assert total_ef == pytest.approx(expected_total_ef, rel=0.05)
