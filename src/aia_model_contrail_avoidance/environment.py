"""Environment module for AIA Model Contrail Avoidance."""

from __future__ import annotations

__all__ = (
    "calculate_effective_radiative_forcing",
    "create_grid_environment",
    "run_flight_data_through_environment",
)

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    import pandas as pd


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
    flight_level_vector = xr.DataArray(
        flight_dataset["flight_level"].values,
    )
    time_vector = xr.DataArray(flight_dataset["timestamp"].values)
    latitude_vector = xr.DataArray(flight_dataset["latitude"].values)
    longitude_vector = xr.DataArray(flight_dataset["longitude"].values)

    nearest_environment = environment.sel(
        level=flight_level_vector,
        time=time_vector,
        latitude=latitude_vector,
        longitude=longitude_vector,
        method="nearest",
    )

    flight_dataset["erf"] = nearest_environment.astype(float)

    return flight_dataset
