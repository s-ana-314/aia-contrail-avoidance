"""Plot histogram of distance traveled in segment from JSON statistics file."""  # noqa: INP001

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def plot_distance_flown_by_altitude_histogram(
    stats_file: str | Path = "2024_01_01_sample_stats_processed",
    output_file: str | Path = "distance_flown_by_altitude_histogram",
) -> None:
    """Plot a histogram of distance flown by flight level from statistics data.

    Args:
        stats_file: Path to the JSON statistics file.
        output_file: Path where the histogram image will be saved.
    """
    with open(f"results/{stats_file}.json") as f:  # noqa: PTH123
        stats = json.load(f)

    # Extract histogram data
    histogram = stats["distance_flown_by_altitude_histogram"]
    bin_edges = np.array(histogram["bin_edges"])
    counts = np.array(histogram["distance_flown"])

    # Calculate bin centers and widths
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    # Handle infinity in the last bin
    if np.isinf(bin_edges[-1]):
        # For the last bin, use a finite value for display purposes
        bin_centers[-1] = bin_edges[-2] + (bin_edges[-2] - bin_edges[-3])
        bin_widths[-1] = bin_edges[-2] - bin_edges[-3]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot bars with appropriate widths
    ax.bar(bin_centers, counts, width=bin_widths, edgecolor="black", alpha=0.7, align="center")

    ax.set_xlabel("Flight Level (FL)", fontsize=12)
    ax.set_ylabel("Distance Flown (m)", fontsize=12)
    ax.set_title(
        "Histogram of Distance Flown at Each Flight Level (FL)", fontsize=14, fontweight="bold"
    )

    ax.grid(axis="y", alpha=0.3, which="both")

    # Add statistics as text
    total_distance = sum(counts)
    ax.text(
        0.98,
        0.98,
        f"Total distance flown: {total_distance:,.2f} m",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    plt.savefig(f"results/plots/{output_file}.png", dpi=300, bbox_inches="tight")
    print(f"Histogram saved as 'results/plots/{output_file}.png'")


if __name__ == "__main__":
    plot_distance_flown_by_altitude_histogram(
        stats_file="2024_01_01_sample_stats_processed",
        output_file="distance_flown_by_altitude_histogram",
    )
