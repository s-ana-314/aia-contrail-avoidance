"""Environment module for AIA Model Contrail Avoidance."""

from __future__ import annotations

__all__ = (
    "calculate_effective_radiative_forcing",
    "create_grid_environment",
    "create_synthetic_grid_environment",
    "run_flight_data_through_environment",
)

import numpy as np
import pandas as pd
import xarray as xr


def calculate_effective_radiative_forcing(
    flight_id: int | list[int], flight_dataset_with_erf: pd.DataFrame
) -> float | list[float]:
    """Calculates total effective radiative forcing for a flight or list of flights."""
    if isinstance(flight_id, int):
        return float(
            flight_dataset_with_erf.loc[
                flight_dataset_with_erf["flight_id"] == flight_id, "erf"
            ].sum()
        )

    total_erf_list = []
    for fid in flight_id:
        total_erf = flight_dataset_with_erf.loc[
            flight_dataset_with_erf["flight_id"] == fid, "erf"
        ].sum()
        total_erf_list.append(total_erf)
    return total_erf_list


def create_grid_environment() -> xr.DataArray:
    """Creates grid environment from COSIP grid data."""
    environment_dataset = xr.open_dataset(
        "./cosip_grid/cocipgrid_sample_result.nc",
        decode_timedelta=True,
        drop_variables=("air_pressure", "altitude", "contrail_age"),
    )
    return xr.DataArray(
        environment_dataset["ef_per_m"], dims=("longitude", "latitude", "level", "time")
    )


def create_synthetic_grid_environment() -> xr.DataArray:
    """Creates a synthetic grid environment for testing."""
    longitudes = xr.DataArray(
        list(range(-8, 3)),  # linear increase from -8 to 2 degrees (UK airspace)
        dims=("longitude"),
    )
    latitudes = xr.DataArray(
        list(range(49, 62)),  # linear increase from 49 to 61 degrees (UK airspace)
        dims=("latitude"),
    )
    levels = xr.DataArray(
        [250, 300, 350],  # flight levels in hPa
        dims=("level"),
    )
    times = xr.DataArray(pd.date_range("2024-01-01", periods=24, freq="1h"), dims=("time"))

    ef_per_m = xr.DataArray(
        np.zeros((len(longitudes), len(latitudes), len(levels), len(times))),
        dims=("longitude", "latitude", "level", "time"),
        coords={
            "longitude": longitudes,
            "latitude": latitudes,
            "level": levels,
            "time": times,
        },
    )

    # Fill with synthetic data
    ef_per_m.loc[{"level": 300}] = 0.5
    ef_per_m.loc[{"level": 350}] = 1.0

    return ef_per_m


def run_flight_data_through_environment(
    flight_dataset: pd.DataFrame, environment: xr.DataArray
) -> pd.DataFrame:
    """Runs flight data through environment to assign effective radiative forcing values.

    Args:
        flight_dataset: DataFrame containing flight data with latitude, longitude, timestamp, and
            flight level.
        environment: xarray DataArray containing environmental data with effective radiative forcing
            values.

    """
    longitude_vector = xr.DataArray(flight_dataset["longitude"].values, dims=["points"])
    latitude_vector = xr.DataArray(flight_dataset["latitude"].values, dims=["points"])
    flight_level_vector = xr.DataArray(
        flight_dataset["flight_level"].values,
        dims=["points"],
    )
    time_vector = xr.DataArray(flight_dataset["timestamp"].values, dims=["points"])

    nearest_environment = environment.sel(
        longitude=longitude_vector,
        latitude=latitude_vector,
        level=flight_level_vector,
        time=time_vector,
        method="nearest",
    )

    flight_dataset["erf"] = nearest_environment.astype(float)

    return flight_dataset
