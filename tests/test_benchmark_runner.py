import asyncio

from llm_serving_lab.benchmark.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
)
from llm_serving_lab.client.vllm_client import (
    GenerationResult,
)
from llm_serving_lab.workload.generator import (
    create_fixed_workload,
)


class FakeVLLMClient:

    async def generate(
        self,
        model,
        prompt,
        max_tokens,
        temperature,
    ):
        await asyncio.sleep(0.01)

        return GenerationResult(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            ttft_ms=2.0,
            latency_ms=10.0,
            inter_token_latency_ms=2.0,
            tokens_per_second=500.0,
        )


def test_benchmark_runner():

    client = FakeVLLMClient()

    runner = BenchmarkRunner(
        client=client
    )

    config = BenchmarkConfig(
        model="test-model",
        concurrency=2,
    )

    workload = create_fixed_workload(
        name="test",
        prompt="hello",
        requests=10,
        max_tokens=128,
    )

    result = asyncio.run(
        runner.run(
            config,
            workload,
        )
    )

    assert len(result.results) == 10
    assert result.requests_per_second > 0
    assert result.output_tokens_per_second > 0