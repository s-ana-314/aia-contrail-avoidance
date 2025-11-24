"""Environment module for AIA Model Contrail Avoidance."""

from __future__ import annotations

__all__ = (
    "calculate_effective_radiative_forcing",
    "create_grid_environment",
    "run_flight_data_through_environment",
)

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
    cocip_grid_dataarray = xr.open_dataset(
        "./cosip_grid/cocipgrid_sample_result.nc", decode_timedelta=True
    ).to_array()
    cocip_grid_dataarray["time"] = pd.to_datetime(cocip_grid_dataarray["time"].values, utc=True)
    # select relevant dimensions
    return cocip_grid_dataarray[
        [
            "longitude",
            "latitude",
            "level",
            "time",
            "ef_per_m",
        ]
    ]


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
    flight_dataset_with_erf = flight_dataset.copy()

    for idx, flight_segment in flight_dataset.iterrows():
        nearest_environment = environment.sel(
            level=flight_segment["flight_level"],
            time=flight_segment["timestamp"],
            latitude=flight_segment["latitude"],
            longitude=flight_segment["longitude"],
            method="nearest",
        )
        flight_dataset_with_erf.loc[idx, "erf"] = float(nearest_environment["ef_per_m"].item())

    return flight_dataset_with_erf
