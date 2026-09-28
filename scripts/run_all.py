"""Run every checked-in KANFIS experiment and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

try:
    from scripts.run_experiment import PROJECT_ROOT, run_experiment
except ModuleNotFoundError:
    from run_experiment import PROJECT_ROOT, run_experiment


SUMMARY_FIELDS = [
    "dataset",
    "task",
    "variant",
    "random_state",
    "rmse",
    "mae",
    "r2",
    "accuracy",
    "macro_f1",
    "output_dir",
]


def run_all(
    config_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Path:
    """Run sorted YAML configurations and return the summary CSV path."""

    configs = Path(config_dir) if config_dir is not None else PROJECT_ROOT / "configs"
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    config_paths = sorted(configs.glob("*.yaml"))
    if not config_paths:
        raise ValueError(f"No YAML configurations found in {configs}.")

    results: list[dict[str, Any]] = []
    for config_path in config_paths:
        results.append(run_experiment(config_path, output_root=root))

    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in SUMMARY_FIELDS})
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    arguments = parser.parse_args()
    print(run_all(arguments.config_dir, arguments.output_root))


if __name__ == "__main__":
    main()
