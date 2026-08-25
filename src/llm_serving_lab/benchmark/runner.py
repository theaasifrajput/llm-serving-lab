from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from llm_serving_lab.client.vllm_client import (
    GenerationResult,
    VLLMClient,
)
from llm_serving_lab.workload.workload import (
    Workload,
    WorkloadRequest,
)


@dataclass
class BenchmarkConfig:
    model: str
    concurrency: int = 1


@dataclass
class BenchmarkResult:
    results: list[GenerationResult]

    total_time_ms: float

    requests_per_second: float
    output_tokens_per_second: float


class BenchmarkRunner:

    def __init__(
        self,
        client: VLLMClient,
    ) -> None:

        self.client = client

    async def _run_request(
        self,
        semaphore: asyncio.Semaphore,
        model: str,
        request: WorkloadRequest,
    ) -> GenerationResult:

        async with semaphore:

            return await self.client.generate(
                model=model,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

    async def run(
        self,
        config: BenchmarkConfig,
        workload: Workload,
    ) -> BenchmarkResult:

        semaphore = asyncio.Semaphore(
            config.concurrency
        )

        start = time.perf_counter()

        tasks = [
            asyncio.create_task(
                self._run_request(
                    semaphore,
                    config.model,
                    request,
                )
            )
            for request in workload.requests
        ]

        results = await asyncio.gather(
            *tasks
        )

        elapsed = (
            time.perf_counter() - start
        )

        total_output_tokens = sum(
            result.output_tokens
            for result in results
        )

        requests_per_second = (
            len(results) / elapsed
            if elapsed > 0
            else 0.0
        )

        output_tokens_per_second = (
            total_output_tokens / elapsed
            if elapsed > 0
            else 0.0
        )

        return BenchmarkResult(
            results=results,
            total_time_ms=elapsed * 1000.0,
            requests_per_second=requests_per_second,
            output_tokens_per_second=output_tokens_per_second,
        )