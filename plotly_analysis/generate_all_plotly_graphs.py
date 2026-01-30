"""This module generates all Plotly graphs in one run."""  # noqa: INP001

from __future__ import annotations

from plotly_contrails_formed_per_time import plot_contrails_formed
from plotly_distance_flown_by_altitude_histogram import plot_distance_flown_by_altitude_histogram
from plotly_energy_forcing_histogram import plot_energy_forcing_histogram
from plotly_uk_airspace import plot_airspace_polygons


def generate_all_plotly() -> None:
    """Generate all Plotly graphs."""
    plot_contrails_formed(
        name_of_forcing_stats_file="energy_forcing_statistics",
        output_plot_name="contrails_formed",
    )
    plot_distance_flown_by_altitude_histogram(
        stats_file="2024_01_01_sample_stats_processed",
        output_file="distance_flown_by_altitude_histogram",
    )
    plot_energy_forcing_histogram(
        json_file="energy_forcing_statistics",
        output_file_histogram="energy_forcing_per_flight_histogram",
        output_file_cumulative="energy_forcing_cumulative",
    )
    plot_airspace_polygons(
        output_file="uk_airspace_map",
    )


if __name__ == "__main__":
    generate_all_plotly()
