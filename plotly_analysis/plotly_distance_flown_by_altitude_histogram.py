"""Plot histogram of distance traveled in segment from JSON statistics file."""  # noqa: INP001

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px  # type: ignore[import-untyped]

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
    total_distance = sum(counts)

    fig = px.bar(
        x=(bin_edges[:-1] + bin_edges[1:]) / 2,
        y=counts,
        labels={"x": "Flight Level", "y": "Distance Flown (meters)"},
        title="Distance Flown by Flight Level Histogram",
    )

    fig.update_layout(
        bargap=0.1,
        modebar_remove=[
            "zoom",
            "pan",
            "select",
            "lasso",
            "zoomIn",
            "zoomOut",
            "autoScale",
            "resetScale",
        ],
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_traces(marker_line_color="black", marker_line_width=1)
    fig.update_xaxes(showline=True, linecolor="black", gridcolor="lightgray", mirror=True)
    fig.update_yaxes(showline=True, linecolor="black", gridcolor="lightgray", mirror=True)

    fig.add_annotation(
        text=f"Total Distance Flown: {total_distance:,.2f} meters",
        xref="paper",
        yref="paper",  # Use relative coordinates (0-1)
        x=0.05,
        y=0.95,  # Top left corner
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )
    fig.update_traces(marker_line_color="black", marker_line_width=1)
    fig.write_html(
        f"results/plots/{output_file}.html",
        config={"displaylogo": False},
        full_html=False,
        include_plotlyjs="cdn",
    )


if __name__ == "__main__":
    plot_distance_flown_by_altitude_histogram(
        stats_file="2024_01_01_sample_stats_processed",
        output_file="distance_flown_by_altitude_histogram",
    )
