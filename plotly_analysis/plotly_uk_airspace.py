from __future__ import annotations  # noqa: D100, INP001

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from aia_model_contrail_avoidance.core_model.airspace import (
    get_gb_airspaces,
)

if TYPE_CHECKING:
    from pathlib import Path


# Define the boundary
SOUTH, NORTH = 39.0, 70.0
WEST, EAST = -12.0, 3.0

# Define the boundary for the graticule grid lines
grid_north = 62.5
grid_south = 45
grid_west = -30
grid_east = 5

# Calculate center from bounds
center_lat = (SOUTH + NORTH) / 2
center_lon = (WEST + EAST) / 2 - 5  # Shift west

# Create a simple map centered on the UK using Maplibre
fig = go.Figure()


def plot_airspace_polygons(
    output_file: str | Path,
) -> None:
    """Plot UK airspace polygons on a map.

    Args:
        output_file: Path to save the output plot image.
    """
    fig.update_layout(
        title="UK Airspace",
        map={
            "style": "carto-voyager",  # Clean, dark style (free, no API key)
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": 2.8,  # Manually tuned to fit the bounds
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        showlegend=False,  # Remove external legend
    )

    # Add latitude lines (graticule)
    for lat in np.arange(grid_south, grid_north + 0.1, 2.5):  # Every 2.5 degrees
        fig.add_trace(
            go.Scattermap(
                lat=[lat] * 100,
                lon=np.linspace(grid_west, grid_east, 100).tolist(),
                mode="lines",
                line={"color": "gray", "width": 0.5},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add longitude lines (graticule)
    for lon in np.arange(grid_west, grid_east + 0.1, 2.5):  # Every 2.5 degrees
        fig.add_trace(
            go.Scattermap(
                lat=np.linspace(grid_south, grid_north, 100).tolist(),
                lon=[lon] * 100,
                mode="lines",
                line={"color": "gray", "width": 0.5},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add longitude labels at the bottom (westings/eastings)
    for lon in np.arange(
        grid_west, grid_east + 0.1, 2.5
    ):  # Labels every 5 degrees within visible area
        label = f"{abs(lon)}°W" if lon < 0 else f"{lon}°E" if lon > 0 else "0°"
        fig.add_trace(
            go.Scattermap(
                lat=[grid_south - 0.5],  # Slightly above bottom edge
                lon=[lon],
                mode="text",
                text=[label],
                textfont={"size": 10, "color": "black"},
                showlegend=False,
                hoverinfo="skip",
            )
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
        )
    # Add latitude labels on the left (northings)
    for lat in np.arange(
        grid_south, grid_north + 0.1, 2.5
    ):  # Labels every 5 degrees within visible area
        label = f"{lat}°N" if lat >= 0 else f"{abs(lat)}°S"
        fig.add_trace(
            go.Scattermap(
                lat=[lat],
                lon=[grid_west - 0.7],  # Slightly right of left edge
                mode="text",
                text=[label],
                textfont={"size": 10, "color": "black"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    uk_airspaces = get_gb_airspaces()

    # Add each airspace as a polygon
    colors = ["rgba(255, 0, 0, 0.2)", "rgba(0, 255, 0, 0.2)", "rgba(0, 0, 255, 0.2)"]
    for i, airspace in enumerate(uk_airspaces):
        # Get the exterior coordinates from the shapely geometry
        coords = np.array(airspace.shape.exterior.coords)
        lons = coords[:, 0]
        lats = coords[:, 1]

        fig.add_trace(
            go.Scattermap(
                lat=lats,
                lon=lons,
                mode="lines",
                fill="toself",
                fillcolor=colors[i % len(colors)],
                line={"color": "black", "width": 1},
                name=getattr(airspace, "name", f"Airspace {i}"),
            )
        )

        # Add label at centroid
        centroid = airspace.shape.centroid
        fig.add_trace(
            go.Scattermap(
                lat=[centroid.y],
                lon=[centroid.x],
                mode="text",
                text=[getattr(airspace, "name", f"Airspace {i}")],
                textfont={"size": 12, "color": "black"},
                showlegend=False,
            )
        )

    fig.write_html(
        f"results/plots/{output_file}.html",
        config={"displaylogo": False, "staticPlot": True},
        full_html=False,
        include_plotlyjs="cdn",
    )


if __name__ == "__main__":
    plot_airspace_polygons(
        output_file="uk_airspace_map",
    )
