from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadRequest:
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


@dataclass(frozen=True)
class Workload:
    name: str
    requests: list[WorkloadRequest]

    @property
    def size(self) -> int:
        return len(self.requests)