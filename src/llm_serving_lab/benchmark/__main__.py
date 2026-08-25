from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from llm_serving_lab.benchmark.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
)
from llm_serving_lab.client.vllm_client import (
    VLLMClient,
)
from llm_serving_lab.metrics.report import (
    create_report,
)
from llm_serving_lab.metrics.serializer import (
    save_report,
)
from llm_serving_lab.workload.generator import (
    create_fixed_workload,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="LLM Serving Lab benchmark"
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--prompt",
        default="Explain how an LLM inference server works.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
    )

    return parser.parse_args()


async def run_benchmark(
    args: argparse.Namespace,
) -> None:

    client = VLLMClient(
        base_url=args.base_url,
    )

    workload = create_fixed_workload(
        name="fixed",
        prompt=args.prompt,
        requests=args.requests,
        max_tokens=args.max_tokens,
    )

    config = BenchmarkConfig(
        model=args.model,
        concurrency=args.concurrency,
    )

    runner = BenchmarkRunner(
        client=client,
    )

    try:

        print()
        print("========== LLM Serving Lab ==========")
        print(f"Model:       {args.model}")
        print(f"Requests:    {args.requests}")
        print(f"Concurrency: {args.concurrency}")
        print(f"Max tokens:  {args.max_tokens}")
        print()

        result = await runner.run(
            config,
            workload,
        )

    finally:

        await client.close()

    report = create_report(result)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_path = (
        Path("results")
        / f"benchmark_{timestamp}.json"
    )

    save_report(
        report,
        output_path,
        model=args.model,
        workload=workload.name,
        concurrency=args.concurrency,
        requests=args.requests,
    )

    print("========== Results ==========")
    print(
        f"Requests/sec: "
        f"{report.requests_per_second:.2f}"
    )

    print(
        f"Output tokens/sec: "
        f"{report.output_tokens_per_second:.2f}"
    )

    print(
        f"TTFT p50: "
        f"{report.ttft.p50:.2f} ms"
    )

    print(
        f"TTFT p95: "
        f"{report.ttft.p95:.2f} ms"
    )

    print(
        f"Latency p50: "
        f"{report.latency.p50:.2f} ms"
    )

    print(
        f"Latency p95: "
        f"{report.latency.p95:.2f} ms"
    )

    print()
    print(f"Results saved to: {output_path}")


def main() -> None:

    args = parse_args()

    asyncio.run(
        run_benchmark(args)
    )


if __name__ == "__main__":
    main()