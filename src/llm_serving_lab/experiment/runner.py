from __future__ import annotations

from dataclasses import dataclass

from llm_serving_lab.benchmark.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
)
from llm_serving_lab.client.vllm_client import (
    VLLMClient,
)
from llm_serving_lab.metrics.report import (
    BenchmarkReport,
    create_report,
)
from llm_serving_lab.workload.config import (
    ExperimentConfig,
)
from llm_serving_lab.workload.generator import (
    create_fixed_workload,
)


@dataclass(frozen=True)
class ExperimentRun:
    concurrency: int


@dataclass(frozen=True)
class ExperimentRunResult:
    concurrency: int
    benchmark: BenchmarkResult
    report: BenchmarkReport


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


async def execute_run(
    config: ExperimentConfig,
    experiment_run: ExperimentRun,
    client: VLLMClient,
) -> ExperimentRunResult:

    model = config.data["model"]["name"]

    workload_config = config.data["workload"]

    workload = create_fixed_workload(
        name=config.name,
        prompt=workload_config["prompt"],
        requests=workload_config["requests"],
        max_tokens=workload_config["max_tokens"],
        temperature=workload_config["temperature"],
    )

    benchmark_config = BenchmarkConfig(
        model=model,
        concurrency=experiment_run.concurrency,
    )

    runner = BenchmarkRunner(
        client=client,
    )

    benchmark_result = await runner.run(
        benchmark_config,
        workload,
    )

    report = create_report(
        benchmark_result
    )

    return ExperimentRunResult(
        concurrency=experiment_run.concurrency,
        benchmark=benchmark_result,
        report=report,
    )
