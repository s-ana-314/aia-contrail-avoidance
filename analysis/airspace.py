"""Plot uk airspace."""  # noqa: INP001

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import shapely
from plot_uk_airspace import generate_uk_airspace_geoaxes
from traffic.data import eurofirs


def get_gb_airspaces() -> list:  # type: ignore[type-arg]
    """Retrieve airspace data for Great Britain FIRs."""
    # Select FIRs by their 'designator' attribute
    selected = []
    for fir in eurofirs:
        if getattr(fir, "designator", None) in ["EGTT", "EGPX", "EGGX"]:
            selected.append(fir)  # noqa: PERF401
    return selected


def plot_airspace_polygons(airspaces: list) -> None:  # type: ignore[type-arg]
    """Plot airspace polygons for given FIRs."""
    uk_airspace_environmental_bounds = {
        "lat_min": 44.0,
        "lat_max": 62.0,
        "lon_min": -35.0,
        "lon_max": 10.0,
    }
    geoax = generate_uk_airspace_geoaxes(uk_airspace_environmental_bounds)

    for fir in airspaces:
        polygon = fir.shape
        if hasattr(polygon, "exterior"):
            x, y = polygon.exterior.xy
            geoax.plot(
                x, y, label=f"{getattr(fir, 'name', '?')} ({getattr(fir, 'designator', '?')})"
            )
        else:
            print(f"No exterior polygon found for {getattr(fir, 'designator', '?')}.")

    geoax.set_xlabel("Longitude")
    geoax.set_ylabel("Latitude")
    geoax.set_title("GB Airspace Polygons")
    geoax.legend()
    geoax.grid(True)  # noqa: FBT003
    plt.tight_layout()
    plt.savefig("results/plots/gb_airspace_polygons.png", dpi=300, bbox_inches="tight")


def find_airspace_of_flight_segment(
    flight_dataframe_path: str,
    airspaces: list,  # type: ignore[type-arg]
) -> pl.DataFrame:
    """Check if a given datapoint is within any of the provided airspaces."""
    flight_dataframe = pl.read_parquet(flight_dataframe_path)

    points = shapely.points(flight_dataframe["longitude"], flight_dataframe["latitude"])

    # add new column indicating the name of the airspace if the point is within any airspace
    return flight_dataframe.with_columns(
        pl.Series("airspace", is_point_in_airspaces(points, airspaces))  # type: ignore[arg-type]
    )


def is_point_in_airspaces(points: list[shapely.Point], airspaces: list) -> list[str | None]:  # type: ignore[type-arg]
    """Check if points are within any of the provided airspaces."""
    names = []
    for point in points:
        found_name = None
        for airspace in airspaces:
            if shapely.contains(airspace.shape, point):
                found_name = getattr(airspace, "name", "?")
                break
        names.append(found_name)
    return names


if __name__ == "__main__":
    gb_airspaces = get_gb_airspaces()
    plot_airspace_polygons(gb_airspaces)

    # Example usage of find_airspace_of_flight_segment
    flight_data_path = (
        "data/contrails_model_data/2024_01_01_sample_processed_with_interpolation.parquet"
    )
    flight_data_with_airspace = find_airspace_of_flight_segment(flight_data_path, gb_airspaces)
    # count distance of points within airspaces
    airspace_counts = flight_data_with_airspace.group_by("airspace").len()
    print(airspace_counts)
    # save the result to a new parquet file
    flight_data_with_airspace.write_parquet("flight_data_with_airspace.parquet")
