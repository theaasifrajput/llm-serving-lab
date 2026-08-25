from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: dict


def load_experiment(
    path: Path,
) -> ExperimentConfig:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Experiment configuration must be a mapping"
        )

    if "name" not in data:
        raise ValueError(
            "Experiment configuration requires 'name'"
        )

    return ExperimentConfig(
        name=data["name"],
        data=data,
    )
