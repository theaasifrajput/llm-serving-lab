from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from llm_serving_lab.client.vllm_client import VLLMClient
from llm_serving_lab.experiment.runner import (
    create_runs,
    execute_run,
)
from llm_serving_lab.metrics.serializer import save_report
from llm_serving_lab.workload.config import load_experiment


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="LLM Serving Lab benchmark"
    )

    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Experiment YAML configuration",
    )

    return parser.parse_args()


async def run_experiment(
    experiment_path: Path,
) -> None:

    config = load_experiment(
        experiment_path
    )

    server_config = config.data["server"]
    base_url = server_config["base_url"]

    model = config.data["model"]["name"]

    runs = create_runs(config)

    client = VLLMClient(
        base_url=base_url
    )

    print()
    print("========== LLM Serving Lab ==========")
    print(f"Experiment:  {config.name}")
    print(f"Model:       {model}")
    print(f"Server:      {base_url}")
    print(f"Runs:        {len(runs)}")
    print()

    try:

        for experiment_run in runs:

            print(
                f"Running concurrency="
                f"{experiment_run.concurrency}"
            )

            result = await execute_run(
                config,
                experiment_run,
                client,
            )

            report = result.report

            timestamp = datetime.now(
                timezone.utc
            ).strftime("%Y%m%dT%H%M%SZ")

            output_path = (
                Path("results")
                / config.name
                / (
                    f"concurrency_"
                    f"{experiment_run.concurrency}_"
                    f"{timestamp}.json"
                )
            )

            save_report(
                report,
                output_path,
                model=model,
                workload=config.name,
                concurrency=experiment_run.concurrency,
                requests=config.data["workload"]["requests"],
            )

            print(
                f"  Requests/sec: "
                f"{report.requests_per_second:.2f}"
            )

            print(
                f"  Output tok/sec: "
                f"{report.output_tokens_per_second:.2f}"
            )

            print(
                f"  TTFT p50: "
                f"{report.ttft.p50:.2f} ms"
            )

            print(
                f"  Results: {output_path}"
            )

            print()

    finally:

        await client.close()


def main() -> None:

    args = parse_args()

    asyncio.run(
        run_experiment(args.experiment)
    )


if __name__ == "__main__":
    main()
