"""Defines a function to plot contrails formed per temporal unit."""  # noqa: INP001

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from aia_model_contrail_avoidance.core_model.dimensions import (
    TemporalGranularity,
    _get_temporal_range_and_labels,
)


def plot_contrails_formed(
    name_of_forcing_stats_file: str,
    output_plot_name: str,
) -> None:
    """Plots the number of contrails formed per temporal unit from the given dataframe.

    Args:
        name_of_flights_stats_file (str): The name of the flights stats file to load data from.
        name_of_forcing_stats_file (str): The name of the forcing stats file to load data from.
        output_plot_name (str): The name of the output plot file.
        temporal_granularity (TemporalGranularity): Granularity for temporal aggregation (default: HOURLY).
    """
    # Load the data from the specified stats file
    with open(f"results/{name_of_forcing_stats_file}.json") as f:  # noqa: PTH123
        forcing_stats_data = json.load(f)

    # read temporal granularity from forcing stats data
    temporal_granularity_str = forcing_stats_data.get("temporal_granularity")
    temporal_granularity = TemporalGranularity(temporal_granularity_str)
    temporal_range, labels = _get_temporal_range_and_labels(temporal_granularity)

    # Extract values from dictionaries
    distance_forming_contrails_per_temporal_histogram = np.array(
        [
            forcing_stats_data.get("distance_forming_contrails_per_temporal_histogram", {}).get(
                str(i), 0
            )
            for i in temporal_range
        ]
    )
    distance_flown_per_temporal_histogram = np.array(
        [
            forcing_stats_data.get("distance_flown_per_temporal_histogram", {}).get(str(i), 0)
            for i in temporal_range
        ]
    )
    air_traffic_density_per_temporal_histogram = np.array(
        [
            forcing_stats_data.get("air_traffic_density_per_temporal_histogram", {}).get(str(i), 0)
            for i in temporal_range
        ]
    )

    percentage_of_distance_forming_contrails = (
        np.divide(
            distance_forming_contrails_per_temporal_histogram,
            distance_flown_per_temporal_histogram,
            out=np.zeros_like(distance_forming_contrails_per_temporal_histogram, dtype=float),
            where=distance_flown_per_temporal_histogram != 0,
        )
        * 100
    )

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot contrails distance on primary y-axis
    ax1.plot(
        range(len(temporal_range)),
        percentage_of_distance_forming_contrails,
        color="skyblue",
        marker="o",
        label="Distance forming contrails",
    )
    ax1.set_xlabel(f"{temporal_granularity.value.capitalize()}")
    ax1.set_ylabel("Percentage of Distance Forming Contrails (%)", color="skyblue")
    ax1.tick_params(axis="y", labelcolor="skyblue")
    ax1.set_yticks(range(0, 11, 1))
    if temporal_granularity == TemporalGranularity.HOURLY:
        ax1.set_xticks(range(len(temporal_range)))
        ax1.set_xticklabels(labels, rotation=20)
    else:
        ax1.set_xticklabels(labels[::30], rotation=45)
        # limit number of x-ticks for daily granularity
        if temporal_granularity == TemporalGranularity.DAILY:
            ticks = range(0, len(temporal_range), 30)
            ax1.set_xticks(ticks)
            ax1.set_xticklabels([labels[i] for i in ticks], rotation=45)

    ax1.grid(axis="y", alpha=0.3)

    # Plot aircraft count on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(
        range(len(temporal_range)),
        air_traffic_density_per_temporal_histogram,
        color="orange",
        marker="s",
        label="Number of aircraft",
    )
    ax2.set_ylabel("Number of Aircraft", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Add title and legend
    plt.title(
        f"Distance Forming Contrails and Air Traffic Density -- {temporal_granularity.value.capitalize()}"
    )

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    output_plot_name = output_plot_name + "_" + temporal_granularity.value + "_plot"

    # Save the plot to the specified output path
    plt.savefig(f"results/plots/{output_plot_name}.png")
    plt.close()


if __name__ == "__main__":
    plot_contrails_formed(
        name_of_forcing_stats_file="energy_forcing_statistics",
        output_plot_name="contrails_formed",
    )
