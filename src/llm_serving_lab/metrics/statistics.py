from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class Statistics:
    count: int

    minimum: float
    maximum: float

    mean: float

    p50: float
    p90: float
    p95: float
    p99: float


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    if not values:
        raise ValueError(
            "Cannot calculate percentile of empty data"
        )

    if not 0 <= percentile_value <= 100:
        raise ValueError(
            "Percentile must be between 0 and 100"
        )

    ordered = sorted(values)

    index = (
        (len(ordered) - 1)
        * percentile_value
        / 100
    )

    lower = int(index)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    weight = index - lower

    return (
        ordered[lower]
        + weight
        * (ordered[upper] - ordered[lower])
    )


def calculate_statistics(
    values: list[float],
) -> Statistics:

    if not values:
        raise ValueError(
            "Cannot calculate statistics from empty data"
        )

    return Statistics(
        count=len(values),

        minimum=min(values),
        maximum=max(values),

        mean=mean(values),

        p50=percentile(values, 50),
        p90=percentile(values, 90),
        p95=percentile(values, 95),
        p99=percentile(values, 99),
    )