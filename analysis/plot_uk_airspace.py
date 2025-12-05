"""Script to visualize UK airspace on a map with specified constraints."""  # noqa: INP001

from __future__ import annotations

from typing import TYPE_CHECKING

import cartopy.crs as ccrs  # type: ignore  # noqa: PGH003
import cartopy.feature as cfeature  # type: ignore  # noqa: PGH003
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes  # type: ignore # noqa: PGH003


# Create a figure with a map projection
def plot_airspace(environmental_bounds: dict[str, float], filename: str) -> None:
    """Plot airspace on a map with specified constraints.

    Args:
        environmental_bounds: Dict with lat_min, lat_max, lon_min, lon_max
        filename: Output filename for the saved figure
    """
    geoax: GeoAxes
    fig, geoax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})

    # Set the extent to show UK airspace
    geoax.set_extent(
        [
            environmental_bounds["lon_min"],
            environmental_bounds["lon_max"],
            environmental_bounds["lat_min"],
            environmental_bounds["lat_max"],
        ],
        crs=ccrs.PlateCarree(),
    )

    # Add map features
    geoax.coastlines(resolution="50m", linewidth=0.5)
    geoax.add_feature(cfeature.BORDERS, linewidth=0.5)
    geoax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.5)
    geoax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    geoax.add_feature(cfeature.LAKES, facecolor="lightblue", alpha=0.5)
    geoax.add_feature(cfeature.RIVERS, linewidth=0.5)

    # Add gridlines
    gl = geoax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    geoax.set_title(
        "UK Airspace - Latitude: 49-62°N, Longitude: -8 to 3°E",
        fontsize=12,
        fontweight="bold",
    )
    geoax.set_xlabel("Longitude (°E)")
    geoax.set_ylabel("Latitude (°N)")

    # Tight layout
    plt.tight_layout()

    # Save and show the figure
    plt.savefig(
        f"results/plots/{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Map saved to results/plots/{filename}.png")


if __name__ == "__main__":
    # Define the constraints
    environmental_bounds = {
        "lat_min": 49.0,
        "lat_max": 62.0,
        "lon_min": -8.0,
        "lon_max": 3.0,
    }
    plot_airspace(environmental_bounds, "uk_airspace_map")
