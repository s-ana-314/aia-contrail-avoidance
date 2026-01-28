"""Plot histogram of distance traveled in segment from JSON statistics file."""  # noqa: INP001

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import plotly.express as px
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

    fig = px.bar(
        x=(bin_edges[:-1] + bin_edges[1:]) / 2,
        y=counts,
        labels={"x": "Flight Level", "y": "Distance Flown (meters)"},
        title="Distance Flown by Flight Level Histogram",
    )

    fig.write_html(f"plotly_analysis/plotly_plots/{output_file}.html")
    #     plt.savefig(f"results/plotly_plots/{output_file}.png", dpi=300, bbox_inches="tight")
    #     print(f"Histogram saved as 'results/plotly_plots/{output_file}.png'")


if __name__ == "__main__":
    plot_distance_flown_by_altitude_histogram(
        stats_file="2024_01_01_sample_stats_processed",
        output_file="distance_flown_by_altitude_histogram",
    )
