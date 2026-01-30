"""Run policy option analysis functions."""  # noqa: INP001

from __future__ import annotations

import polars as pl
from generate_energy_forcing_statistics import generate_energy_forcing_statistics

from aia_model_contrail_avoidance.policy import (
    ContrailAvoidancePolicy,
    apply_contrail_avoidance_policy,
)

# Read in ADS-B datafrmaw from parquet file
parquet_filename = "2024_01_01_sample_processed_with_interpolation_with_ef"
complete_flight_dataframe = pl.read_parquet(f"data/contrails_model_data/{parquet_filename}.parquet")


selected_dataframe = apply_contrail_avoidance_policy(
    ContrailAvoidancePolicy.AVOID_ALL_CONTRAILS_AT_NIGHT_IN_UK_AIRSPACE,
    complete_flight_dataframe,
)

save_path = "policy_data/2024_01_01_sample_processed_with_policy_avoid_all_contrails_at_night_in_uk_airspace.parquet"
generate_energy_forcing_statistics(selected_dataframe, save_path)
