"""Run one YAML-configured KANFIS experiment on a built-in dataset."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.datasets import load_diabetes, load_iris
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from kanfis import KANFISClassifier, KANFISRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_KEYS = {
    "dataset",
    "task",
    "variant",
    "test_size",
    "random_state",
    "model",
    "training",
}
MODEL_KEYS = {"n_rules", "n_memberships", "n_layers", "membership"}
TRAINING_KEYS = {
    "learning_rate",
    "batch_size",
    "max_epochs",
    "validation_fraction",
    "patience",
    "min_delta",
    "lambda_entropy",
    "lambda_distinct",
    "weight_decay",
    "device",
    "verbose",
}


def _require_exact_keys(mapping: dict[str, Any], expected: set[str], section: str) -> None:
    missing = expected - set(mapping)
    unknown = set(mapping) - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ValueError(f"Invalid {section} keys: {', '.join(details)}.")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate an experiment YAML document."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")
    _require_exact_keys(config, TOP_LEVEL_KEYS, "top-level")
    if not isinstance(config["model"], dict) or not isinstance(config["training"], dict):
        raise ValueError("model and training sections must be YAML mappings.")
    _require_exact_keys(config["model"], MODEL_KEYS, "model")
    _require_exact_keys(config["training"], TRAINING_KEYS, "training")

    if config["dataset"] not in {"diabetes", "iris"}:
        raise ValueError("dataset must be either 'diabetes' or 'iris'.")
    expected_task = "regression" if config["dataset"] == "diabetes" else "classification"
    if config["task"] != expected_task:
        raise ValueError(
            f"Dataset {config['dataset']!r} requires task={expected_task!r}."
        )
    if config["variant"] not in {"it1", "it2"}:
        raise ValueError("variant must be either 'it1' or 'it2'.")
    if not 0 < float(config["test_size"]) < 1:
        raise ValueError("test_size must be between 0 and 1.")
    return config


def _load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if name == "diabetes":
        dataset = load_diabetes()
    elif name == "iris":
        dataset = load_iris()
    else:
        raise ValueError("Unknown built-in dataset.")
    return dataset.data, dataset.target, list(dataset.feature_names)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _scaler_state(scaler: Any) -> dict[str, Any]:
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": _json_value(scaler.n_samples_seen_),
    }


def _checkpoint(estimator: Any) -> dict[str, Any]:
    checkpoint = {
        "estimator": estimator.__class__.__name__,
        "parameters": estimator.get_params(),
        "network_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in estimator.network_.state_dict().items()
        },
        "x_scaler": _scaler_state(estimator.x_scaler_),
        "feature_names": estimator.feature_names_in_.tolist(),
        "n_features_in": int(estimator.n_features_in_),
        "resolved_device": str(estimator.device_),
    }
    if isinstance(estimator, KANFISRegressor):
        checkpoint["y_scaler"] = _scaler_state(estimator.y_scaler_)
    else:
        checkpoint["classes"] = _json_value(estimator.classes_)
    return checkpoint


def _environment(estimator: Any) -> dict[str, Any]:
    """Record the runtime versions and resolved compute device."""

    import sklearn

    return {
        "python": platform.python_version(),
        "packages": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "pyyaml": yaml.__version__,
        },
        "resolved_device": str(estimator.device_),
    }


def run_experiment(
    config_path: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one experiment and return metrics plus the artifact directory."""

    config = load_config(config_path)
    features, target, feature_names = _load_dataset(config["dataset"])
    stratify = target if config["task"] == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=float(config["test_size"]),
        random_state=int(config["random_state"]),
        shuffle=True,
        stratify=stratify,
    )

    estimator_class = (
        KANFISRegressor if config["task"] == "regression" else KANFISClassifier
    )
    estimator_parameters = {
        "variant": config["variant"],
        "random_state": config["random_state"],
        **config["model"],
        **config["training"],
    }
    estimator = estimator_class(**estimator_parameters)
    estimator.fit(x_train, y_train)
    prediction = estimator.predict(x_test)

    metrics: dict[str, Any] = {
        "dataset": config["dataset"],
        "task": config["task"],
        "variant": config["variant"],
        "random_state": int(config["random_state"]),
    }
    if config["task"] == "regression":
        metrics.update(
            {
                "rmse": math.sqrt(mean_squared_error(y_test, prediction)),
                "mae": mean_absolute_error(y_test, prediction),
                "r2": r2_score(y_test, prediction),
            }
        )
    else:
        metrics.update(
            {
                "accuracy": accuracy_score(y_test, prediction),
                "macro_f1": f1_score(y_test, prediction, average="macro"),
            }
        )

    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    output_dir = (
        root
        / str(config["dataset"])
        / str(config["variant"])
        / f"seed_{config['random_state']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = copy.deepcopy(config)
    resolved_config["training"]["device"] = str(estimator.device_)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "history.json", estimator.history_)
    _write_json(output_dir / "environment.json", _environment(estimator))
    estimator.export_rules(output_dir / "rules.json", feature_names=feature_names)
    estimator.export_rules(output_dir / "rules.txt", feature_names=feature_names)
    torch.save(_checkpoint(estimator), output_dir / "model.pt")

    result = dict(metrics)
    result["output_dir"] = str(output_dir.relative_to(root))
    return _json_value(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    arguments = parser.parse_args()
    result = run_experiment(arguments.config, arguments.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
