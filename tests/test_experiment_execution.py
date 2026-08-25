import asyncio

from llm_serving_lab.client.vllm_client import GenerationResult
from llm_serving_lab.experiment.runner import (
    execute_run,
    ExperimentRun,
)
from llm_serving_lab.workload.config import (
    ExperimentConfig,
)


class FakeVLLMClient:

    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:

        return GenerationResult(
            text="fake response",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            ttft_ms=10.0,
            latency_ms=50.0,
            inter_token_latency_ms=2.0,
            tokens_per_second=400.0,
        )


def test_execute_run():

    config = ExperimentConfig(
        name="test-experiment",
        data={
            "model": {
                "name": "test-model",
            },
            "workload": {
                "prompt": "hello",
                "requests": 3,
                "max_tokens": 32,
                "temperature": 0.0,
            },
            "benchmark": {
                "concurrency": 2,
            },
        },
    )

    run = ExperimentRun(
        concurrency=2,
    )

    result = asyncio.run(
        execute_run(
            config=config,
            experiment_run=run,
            client=FakeVLLMClient(),
        )
    )

    assert result.concurrency == 2

    assert len(
        result.benchmark.results
    ) == 3

    assert (
        result.report.ttft.mean
        == 10.0
    )

    assert (
        result.report.latency.mean
        == 50.0
    )

    assert (
        result.report.tokens_per_second.mean
        == 400.0
    )
