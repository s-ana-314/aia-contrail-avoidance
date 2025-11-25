"""Tests for generating environment and erf data."""

from __future__ import annotations

import datetime

from aia_model_contrail_avoidance.environment import (
    create_grid_environment,
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


def test_run_flight_data_through_environment() -> None:
    """Test running flight data through environment."""
    environment = create_grid_environment()
    departure_time = datetime.datetime(2022, 3, 1, 1, 0, 0, tzinfo=datetime.UTC)
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
