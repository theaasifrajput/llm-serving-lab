from pathlib import Path

import pytest

from llm_serving_lab.experiment.runner import (
    create_runs,
    get_concurrency_values,
)
from llm_serving_lab.workload.config import (
    load_experiment,
)


def test_concurrency_sweep():

    config = load_experiment(
        Path("experiments/concurrency.yaml")
    )

    values = get_concurrency_values(config)

    assert values == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]


def test_create_runs():

    config = load_experiment(
        Path("experiments/concurrency.yaml")
    )

    runs = create_runs(config)

    assert len(runs) == 6

    assert [
        run.concurrency
        for run in runs
    ] == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]


def test_invalid_concurrency():

    config = load_experiment(
        Path("experiments/concurrency.yaml")
    )

    config.data["benchmark"]["concurrency"] = "16"

    with pytest.raises(ValueError):
        get_concurrency_values(config)
