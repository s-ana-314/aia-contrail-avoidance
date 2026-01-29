"""Enums for spatial and temporal granularity in the contrail avoidance model."""

from __future__ import annotations

__all__ = [
    "SpatialGranularity",
    "TemporalGranularity",
    "_get_temporal_grouping_field",
    "_get_temporal_range_and_labels",
]
from enum import Enum

import inquirer  # type: ignore[import-untyped]


class SpatialGranularity(Enum):
    """Defines spatial granularity options for airspace partitioning."""

    UK_AIRSPACE = "uk_airspace"
    ONE_DEGREE = "one_degree"


class TemporalGranularity(Enum):
    """Defines temporal granularity options for time aggregation."""

    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    SEASONALLY = "seasonally"
    ANNUALLY = "annually"


def _get_temporal_grouping_field(temporal_granularity: TemporalGranularity) -> str:
    """Get the temporal grouping field based on granularity.

    Args:
        temporal_granularity: The temporal granularity to use.

    Returns:
        The polars datetime method name to extract the appropriate time unit.
    """
    mapping = {
        TemporalGranularity.HOURLY: "hour",
        TemporalGranularity.DAILY: "day",
        TemporalGranularity.MONTHLY: "month",
        TemporalGranularity.SEASONALLY: "quarter",
        TemporalGranularity.ANNUALLY: "year",
    }
    return mapping[temporal_granularity]


def _get_temporal_range_and_labels(
    temporal_granularity: TemporalGranularity,
) -> tuple[range, list[str]]:
    """Get the range and labels for temporal granularity.

    Args:
        temporal_granularity: The temporal granularity to use.

    Returns:
        Tuple of (range of values, list of labels for x-axis).
    """
    if temporal_granularity == TemporalGranularity.HOURLY:
        return range(24), [f"{i:02d}:00" for i in range(24)]
    if temporal_granularity == TemporalGranularity.DAILY:
        return range(1, 366), [str(i) for i in range(1, 366)]
    if temporal_granularity == TemporalGranularity.MONTHLY:
        return range(1, 13), [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    if temporal_granularity == TemporalGranularity.SEASONALLY:
        return range(1, 5), ["Q1", "Q2", "Q3", "Q4"]
    if temporal_granularity == TemporalGranularity.ANNUALLY:
        return range(2024, 2025), ["Year"]
    msg = f"Unknown temporal granularity: {temporal_granularity}"
    raise ValueError(msg)
    return None


def user_input_temporal_granularity() -> TemporalGranularity:
    questions = [
        inquirer.List(
            "Time_Scale",
            message="What time scale do you want to process",
            choices=["HOURLY", "DAILY", "MONTHLY", "SEASONALLY", "ANNUALLY"],
        ),
    ]
    answers = inquirer.prompt(questions)
    temporal = TemporalGranularity[answers["Time_Scale"]]
    print(temporal)
    return temporal
