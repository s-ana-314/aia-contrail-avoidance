"""Process ADS-B flight data from parquet file."""  # noqa: INP001

from __future__ import annotations

import time

from aia_model_contrail_avoidance.flight_data_processing import (
    FlightDepartureAndArrivalSubset,
    TemporalFlightSubset,
    process_ads_b_flight_data,
)

if __name__ == "__main__":
    start = time.time()
    parquet_file_path = "data/flight_data/2024_01_01_sample.parquet"
    save_filename = "2024_01_01_sample_processed_with_interpolation"
    temporal_flight_subset = TemporalFlightSubset.FIRST_MONTH
    flight_departure_and_arrival = FlightDepartureAndArrivalSubset.UK

    process_ads_b_flight_data(
        parquet_file_path,
        save_filename,
        flight_departure_and_arrival,
        temporal_flight_subset,
    )
    end = time.time()
    length = end - start
    print("Data processing completed in", round(length / 60, 1), "minutes.")
