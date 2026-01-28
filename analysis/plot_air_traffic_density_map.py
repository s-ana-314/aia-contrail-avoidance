"""Plot air traffic density as a matrix for each degree using SpatialGranularity enum."""  # noqa: INP001

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from plot_uk_airspace import generate_uk_airspace_geoaxes

from aia_model_contrail_avoidance.core_model.dimensions import SpatialGranularity


def plot_air_traffic_density_matrix(  # noqa: PLR0915
    parquet_file_name: str,
    environmental_bounds: dict[str, float] | None = None,
    spatial_granularity: SpatialGranularity = SpatialGranularity.ONE_DEGREE,
    output_plot_name: str = "air_traffic_density_map",
) -> None:
    """Plot air traffic density as a heatmap matrix for each degree.

    Args:
        parquet_file_name: Name of the parquet file containing flight data (without extension).
        spatial_granularity: Spatial granularity for binning (default: ONE_DEGREE).
        environmental_bounds: Optional dict with lat_min, lat_max, lon_min, lon_max.
        output_plot_name: Name of the output plot file (without extension).

    Raises:
        FileNotFoundError: If the parquet file is not found.
        NotImplementedError: If the chosen spatial granularity is not supported.
    """
    # Load flight data
    parquet_file_path = Path("data/contrails_model_data") / f"{parquet_file_name}.parquet"
    if not parquet_file_path.exists():
        msg = f"Parquet file not found: {parquet_file_path}"
        raise FileNotFoundError(msg)

    flight_dataframe = pl.read_parquet(parquet_file_path)

    # Create spatial bins and compute air traffic density
    if spatial_granularity == SpatialGranularity.ONE_DEGREE:
        # Create degree bins for latitude and longitude
        flight_dataframe_with_bins = flight_dataframe.with_columns(
            [
                pl.col("latitude").floor().alias("lat_bin"),
                pl.col("longitude").floor().alias("lon_bin"),
            ]
        )

        # Count unique flights per degree bin (air traffic density)
        density_data = (
            flight_dataframe_with_bins.group_by(["lat_bin", "lon_bin"])
            .agg(pl.col("flight_id").n_unique().alias("flight_count"))
            .sort(["lon_bin", "lat_bin"])
        )

        # Set environmental bounds
        if environmental_bounds is None:
            min_lat = int(flight_dataframe_with_bins["lat_bin"].min())  # type: ignore[arg-type]
            max_lat = int(flight_dataframe_with_bins["lat_bin"].max())  # type: ignore[arg-type]
            min_lon = int(flight_dataframe_with_bins["lon_bin"].min())  # type: ignore[arg-type]
            max_lon = int(flight_dataframe_with_bins["lon_bin"].max())  # type: ignore[arg-type]
            environmental_bounds = {
                "lat_min": min_lat,
                "lat_max": max_lat,
                "lon_min": min_lon,
                "lon_max": max_lon,
            }
        else:
            min_lat = int(environmental_bounds["lat_min"])
            max_lat = int(environmental_bounds["lat_max"])
            min_lon = int(environmental_bounds["lon_min"])
            max_lon = int(environmental_bounds["lon_max"])

        # Create matrix with zeros
        lat_range = max_lat - min_lat + 1
        lon_range = max_lon - min_lon + 1
        density_matrix = np.zeros((lat_range, lon_range))

        # Populate matrix and hande out-of-bounds gracefully
        for row in density_data.iter_rows():
            lat_bin = int(row[0])
            lon_bin = int(row[1])
            flight_count = row[2]

            if min_lat <= lat_bin <= max_lat and min_lon <= lon_bin <= max_lon:
                lat_index = lat_bin - min_lat
                lon_index = lon_bin - min_lon
                density_matrix[lat_index, lon_index] = flight_count

    elif spatial_granularity == SpatialGranularity.UK_AIRSPACE:
        # Use UK airspace bounds
        if environmental_bounds is None:
            environmental_bounds = {
                "lat_min": 49.0,
                "lat_max": 62.0,
                "lon_min": -8.0,
                "lon_max": 3.0,
            }

        min_lat = int(environmental_bounds["lat_min"])
        max_lat = int(environmental_bounds["lat_max"])
        min_lon = int(environmental_bounds["lon_min"])
        max_lon = int(environmental_bounds["lon_max"])

        # Create UK airspace bins  handle out-of-bounds gracefully
        flight_dataframe_in_uk = flight_dataframe.filter(
            (pl.col("latitude") >= min_lat)
            & (pl.col("latitude") <= max_lat)
            & (pl.col("longitude") >= min_lon)
            & (pl.col("longitude") <= max_lon)
        )
        density_result = flight_dataframe_in_uk.select(
            pl.col("flight_id").n_unique().alias("flight_count")
        )
        flight_count = int(density_result[0, "flight_count"])
        density_matrix = np.array([[flight_count]])
    else:
        msg = f"Spatial granularity '{spatial_granularity}' not supported."
        raise NotImplementedError(msg)

    # Create figure with map projection
    geoax = generate_uk_airspace_geoaxes(environmental_bounds=environmental_bounds)

    # Plot heatmap overlay on map
    im = geoax.imshow(
        density_matrix[::-1],  # Flip to have north at top
        extent=[min_lon, max_lon + 1, min_lat, max_lat + 1],
        cmap="YlOrRd",
        aspect="auto",
        origin="lower",
        transform=ccrs.PlateCarree(),
        alpha=0.7,  # Semi-transparent to see map features underneath
    )

    # Add gridlines
    gl = geoax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # Labels and title
    geoax.set_title(
        "Air Traffic Density per Degree (1° x 1° Grid) - UK Airspace",
        fontsize=14,
        fontweight="bold",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=geoax, orientation="vertical", pad=0.1)
    cbar.set_label("Number of Unique Flights", fontsize=11, fontweight="bold")

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = Path("results/plots") / f"{output_plot_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    plt.close()


if __name__ == "__main__":
    environmental_bounds = {
        "lat_min": 49.0,
        "lat_max": 62.0,
        "lon_min": -8.0,
        "lon_max": 3.0,
    }
    plot_air_traffic_density_matrix(
        parquet_file_name="2024_01_01_sample_processed_with_interpolation",
        environmental_bounds=environmental_bounds,
        spatial_granularity=SpatialGranularity.UK_AIRSPACE,
        output_plot_name="air_traffic_density_map_uk_airspace",
    )
