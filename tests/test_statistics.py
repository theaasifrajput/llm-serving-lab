import pytest

from llm_serving_lab.metrics.statistics import (
    calculate_statistics,
    percentile,
)


def test_percentile():

    values = [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ]

    assert percentile(values, 0) == 10.0
    assert percentile(values, 50) == 30.0
    assert percentile(values, 100) == 50.0


def test_statistics():

    values = [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ]

    stats = calculate_statistics(values)

    assert stats.count == 5
    assert stats.minimum == 10.0
    assert stats.maximum == 50.0
    assert stats.mean == 30.0

    assert stats.p50 == 30.0


def test_empty_values():

    with pytest.raises(ValueError):
        calculate_statistics([])


def test_invalid_percentile():

    with pytest.raises(ValueError):
        percentile(
            [1.0, 2.0, 3.0],
            101,
        )