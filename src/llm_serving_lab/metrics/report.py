from __future__ import annotations

from dataclasses import dataclass

from llm_serving_lab.benchmark.runner import BenchmarkResult
from llm_serving_lab.metrics.statistics import (
    Statistics,
    calculate_statistics,
)


@dataclass(frozen=True)
class BenchmarkReport:
    ttft: Statistics
    latency: Statistics
    inter_token_latency: Statistics
    tokens_per_second: Statistics

    requests_per_second: float
    output_tokens_per_second: float


def create_report(
    result: BenchmarkResult,
) -> BenchmarkReport:

    ttft = [
        item.ttft_ms
        for item in result.results
    ]

    latency = [
        item.latency_ms
        for item in result.results
    ]

    inter_token_latency = [
        item.inter_token_latency_ms
        for item in result.results
    ]

    tokens_per_second = [
        item.tokens_per_second
        for item in result.results
    ]

    return BenchmarkReport(
        ttft=calculate_statistics(ttft),
        latency=calculate_statistics(latency),
        inter_token_latency=calculate_statistics(
            inter_token_latency
        ),
        tokens_per_second=calculate_statistics(
            tokens_per_second
        ),
        requests_per_second=result.requests_per_second,
        output_tokens_per_second=result.output_tokens_per_second,
    )
