from __future__ import annotations

from llm_serving_lab.workload.workload import (
    Workload,
    WorkloadRequest,
)


def create_fixed_workload(
    name: str,
    prompt: str,
    requests: int,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> Workload:

    workload_requests = [
        WorkloadRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for _ in range(requests)
    ]

    return Workload(
        name=name,
        requests=workload_requests,
    )