from __future__ import annotations

from dataclasses import dataclass

from llm_serving_lab.workload.config import ExperimentConfig


@dataclass(frozen=True)
class ExperimentRun:
    concurrency: int


def get_concurrency_values(
    config: ExperimentConfig,
) -> list[int]:

    value = config.data["benchmark"]["concurrency"]

    if isinstance(value, int):
        return [value]

    if isinstance(value, list):
        if not value:
            raise ValueError(
                "Concurrency list cannot be empty"
            )

        if not all(
            isinstance(item, int)
            for item in value
        ):
            raise ValueError(
                "Concurrency values must be integers"
            )

        return value

    raise ValueError(
        "benchmark.concurrency must be "
        "an integer or list of integers"
    )


def create_runs(
    config: ExperimentConfig,
) -> list[ExperimentRun]:

    return [
        ExperimentRun(
            concurrency=value
        )
        for value in get_concurrency_values(config)
    ]
