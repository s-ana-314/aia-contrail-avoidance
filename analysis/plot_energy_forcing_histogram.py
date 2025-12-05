"""Plot histogram of energy forcing per flight with cumulative forcing analysis."""  # noqa: INP001

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_energy_forcing_histogram(json_file: str, output_plot: str) -> None:
    """Plot histogram of energy forcing per flight with cumulative forcing analysis.

    Args:
        json_file: Path to the JSON file containing energy forcing statistics
        output_plot: Path to save the output plot image
    """
    # Load the JSON file
    with open(f"results/{json_file}.json") as f:  # noqa: PTH123
        stats = json.load(f)

    # Load the full flight data with energy forcing
    flight_data = pd.read_parquet("data/2024_01_01_sample_with_ef.parquet")

    # Calculate total energy forcing per flight
    flight_ef_summary = flight_data.groupby("flight_id")["ef"].sum().reset_index(name="total_ef")
    flight_ef_summary = flight_ef_summary.sort_values("total_ef", ascending=False).reset_index(
        drop=True
    )

    # Calculate cumulative energy forcing
    flight_ef_summary["cumulative_ef"] = flight_ef_summary["total_ef"].cumsum()
    total_energy_forcing = flight_ef_summary["total_ef"].sum()

    # Find how many flights contribute to 80%, 50%, and 20% of total forcing
    ef_80_percent = total_energy_forcing * 0.8
    ef_50_percent = total_energy_forcing * 0.5
    ef_20_percent = total_energy_forcing * 0.2

    flights_for_80_percent = (flight_ef_summary["cumulative_ef"] <= ef_80_percent).sum()
    flights_for_50_percent = (flight_ef_summary["cumulative_ef"] <= ef_50_percent).sum()
    flights_for_20_percent = (flight_ef_summary["cumulative_ef"] <= ef_20_percent).sum()

    # Extract histogram data for plotting
    histogram = stats["energy_forcing_per_flight"]["histogram"]
    bin_edges = np.array(histogram["bin_edges"])
    counts = np.array(histogram["counts"])

    total_flights = len(flight_ef_summary)

    # Calculate bin centers and widths
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # --- Top subplot: Regular histogram ---
    ax1.bar(bin_centers, counts, width=bin_widths, edgecolor="black", alpha=0.7, align="center")

    ax1.set_xlabel("Total Energy Forcing per Flight (J)", fontsize=12)
    ax1.set_ylabel("Number of Flights", fontsize=12)
    ax1.set_title("Distribution of Energy Forcing per Flight", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Add text box with statistics
    ax1.text(
        0.98,
        0.98,
        f"Total flights: {total_flights:,}\n"
        f"Total EF: {total_energy_forcing:.2e} J\n\n"
        f"20% of forcing: {flights_for_20_percent:,} flights ({flights_for_20_percent / total_flights * 100:.1f}%)\n"
        f"50% of forcing: {flights_for_50_percent:,} flights ({flights_for_50_percent / total_flights * 100:.1f}%)\n"
        f"80% of forcing: {flights_for_80_percent:,} flights ({flights_for_80_percent / total_flights * 100:.1f}%)",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # --- Bottom subplot: Cumulative energy forcing ---
    flight_indices = np.arange(1, len(flight_ef_summary) + 1)
    cumulative_ef_percentage = (flight_ef_summary["cumulative_ef"] / total_energy_forcing) * 100

    ax2.plot(flight_indices, cumulative_ef_percentage, linewidth=2, color="blue")
    ax2.axhline(
        80,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"80% of forcing ({flights_for_80_percent} flights)",
    )
    ax2.axhline(
        50,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"50% of forcing ({flights_for_50_percent} flights)",
    )
    ax2.axhline(
        20,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"20% of forcing ({flights_for_20_percent} flights)",
    )

    ax2.axvline(flights_for_80_percent, color="green", linestyle=":", linewidth=1.5, alpha=0.7)
    ax2.axvline(flights_for_50_percent, color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
    ax2.axvline(flights_for_20_percent, color="red", linestyle=":", linewidth=1.5, alpha=0.7)

    ax2.set_xlabel("Number of Flights (sorted by EF, highest first)", fontsize=12)
    ax2.set_ylabel("Cumulative Energy Forcing (%)", fontsize=12)
    ax2.set_title("Cumulative Energy Forcing Contribution", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, total_flights)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(f"results/plots/{output_plot}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    input_json = "energy_forcing_statistics"
    output_plot = "energy_forcing_per_flight_histogram"
    plot_energy_forcing_histogram(input_json, output_plot)
