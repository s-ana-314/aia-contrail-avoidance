"""Defines a function to plot contrails formed per temporal unit."""  # noqa: INP001

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

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
    time_label = "Time" if temporal_granularity == TemporalGranularity.HOURLY else "Day"

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

    # Plot contrails distance on primary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(temporal_range)),
            y=percentage_of_distance_forming_contrails,
            mode="lines+markers",
            name="Percentage of Distance Forming Contrails",
            line={"color": "blue"},
            marker={"color": "blue"},
            customdata=labels,
            hovertemplate=(
                f"{time_label}: %{{customdata}}<br>"
                "Percent of distance forming contrails: %{y:.2f}%"
            ),
        ),
        secondary_y=False,
    )

    # Plot air traffic density on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(temporal_range)),
            y=air_traffic_density_per_temporal_histogram,
            mode="lines+markers",
            name="Air Traffic Density",
            line={"color": "red"},
            marker={"color": "red"},
            customdata=labels,
            hovertemplate=(f"{time_label}: %{{customdata}}<br>Aircraft: %{{y:.0f}}<extra></extra>"),
        ),
        secondary_y=True,
    )

    fig.update_traces(
        marker={"size": 8},
    )

    fig.update_layout(
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
        title="Distance Forming Contrails and Air Traffic Density --"
        f" {temporal_granularity.value.capitalize()}",
        legend={
            "x": 0.90,
            "y": 0.99,
            "xanchor": "right",
            "yanchor": "top",
            "font": {"size": 12},
            "itemsizing": "constant",
            "itemwidth": 30,
            "bgcolor": "rgba(255, 255, 255, 0.7)",
            "bordercolor": "lightgray",
            "borderwidth": 1,
        },
    )

    fig.update_xaxes(
        showgrid=True,
        showline=True,
        linecolor="black",
        gridcolor="lightgray",
        zeroline=True,
        zerolinecolor="lightgray",
        zerolinewidth=1,
        mirror=True,
    )
    fig.update_yaxes(
        title_text="Percentage of Distance\\ Forming Contrails (%)",
        title_font={"color": "blue"},
        rangemode="tozero",
        showline=True,
        linecolor="black",
        gridcolor="lightgray",
        mirror=True,
        secondary_y=False,
    )
    fig.update_yaxes(
        rangemode="tozero",
        title_text="Number of Aircraft",
        title_font={"color": "red"},
        showgrid=False,
        mirror=True,
        secondary_y=True,
    )

    if temporal_granularity == TemporalGranularity.HOURLY:
        fig.update_xaxes(
            title_text=f"Time--{temporal_granularity.value.capitalize()}",
            tickmode="array",
            tickvals=list(range(len(temporal_range))),
            ticktext=labels,
        )
    elif temporal_granularity == TemporalGranularity.DAILY:
        daily_tick_indices = [0, *list(range(29, len(temporal_range), 30))]
        fig.update_xaxes(
            title_text="Day of Year",
            tickmode="array",
            tickvals=daily_tick_indices,
            ticktext=[labels[i] for i in daily_tick_indices],
        )

    # Save the plot to the specified output path
    fig.write_html(
        f"results/plots/{output_plot_name}.html",
        config={"displaylogo": False},
        full_html=False,
        include_plotlyjs="cdn",
    )


if __name__ == "__main__":
    plot_contrails_formed(
        name_of_forcing_stats_file="energy_forcing_statistics",
        output_plot_name="contrails_formed",
    )
