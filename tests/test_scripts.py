import csv
import json
from pathlib import Path

import pytest
import yaml

from scripts.run_experiment import PROJECT_ROOT, load_config, run_experiment
from scripts.run_all import run_all


@pytest.mark.parametrize(
    "config_name",
    ["diabetes_it1.yaml", "diabetes_it2.yaml", "iris_it1.yaml", "iris_it2.yaml"],
)
def test_checked_in_configs_are_complete_and_use_cpu(config_name):
    config = load_config(PROJECT_ROOT / "configs" / config_name)

    assert config["training"]["device"] == "cpu"


def test_run_experiment_writes_declared_artifacts(tmp_path):
    config = {
        "dataset": "diabetes",
        "task": "regression",
        "variant": "it1",
        "test_size": 0.2,
        "random_state": 42,
        "model": {
            "n_rules": 2,
            "n_memberships": 2,
            "n_layers": 1,
            "membership": "gaussian",
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 64,
            "max_epochs": 1,
            "validation_fraction": 0.2,
            "patience": 1,
            "min_delta": 0.0001,
            "lambda_entropy": 0.0,
            "lambda_distinct": 0.0,
            "weight_decay": 0.01,
            "device": "cpu",
            "verbose": 0,
        },
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_experiment(config_path, output_root=tmp_path / "outputs")
    output_dir = tmp_path / "outputs" / "diabetes" / "it1" / "seed_42"

    assert {
        "config.yaml",
        "metrics.json",
        "history.json",
        "rules.json",
        "rules.txt",
        "model.pt",
        "environment.json",
    } == {path.name for path in output_dir.iterdir()}
    assert result["output_dir"] == str(Path("diabetes") / "it1" / "seed_42")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {"rmse", "mae", "r2"}.issubset(metrics)


def test_run_all_writes_summary_with_relative_artifact_path(tmp_path):
    config = load_config(PROJECT_ROOT / "configs" / "diabetes_it1.yaml")
    config["model"]["n_rules"] = 2
    config["training"]["max_epochs"] = 1
    config["training"]["patience"] = 1
    config["training"]["verbose"] = 0
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "experiment.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )

    summary_path = run_all(config_dir, tmp_path / "outputs")

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["output_dir"] == str(Path("diabetes") / "it1" / "seed_42")
