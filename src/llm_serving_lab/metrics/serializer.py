from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from llm_serving_lab.metrics.report import BenchmarkReport


def save_report(
    report: BenchmarkReport,
    path: Path,
    *,
    model: str,
    workload: str,
    concurrency: int,
    requests: int,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "model": model,
        "workload": workload,
        "concurrency": concurrency,
        "requests": requests,
        "metrics": asdict(report),
    }

    path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )