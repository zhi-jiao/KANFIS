"""Extract learned KANFIS parameters as structured, readable rules."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ._validation import validate_feature_names


def _label_name(label: Any) -> str:
    if isinstance(label, np.generic):
        label = label.item()
    return str(label)


def extract_rules(
    network: nn.Module,
    feature_names: Sequence[str] | None = None,
    class_labels: Sequence[Any] | None = None,
    threshold: float = 0.0,
    feature_offsets: Sequence[float] | None = None,
    feature_scales: Sequence[float] | None = None,
    output_offsets: Sequence[float] | None = None,
    output_scales: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Return one JSON-safe parameter record for each first-layer rule."""

    if not np.isscalar(threshold) or not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("threshold must be a non-negative finite number.")
    if not hasattr(network, "layers") or not network.layers:
        raise ValueError("network must expose at least one KANFIS layer.")
    if not hasattr(network, "final_layer"):
        raise ValueError("network must expose a final_layer.")
    if int(getattr(network, "n_layers", 1)) != 1:
        raise ValueError(
            "Rule extraction requires n_layers=1 because deeper networks do not "
            "have direct first-layer consequents."
        )

    layer = network.layers[0]
    n_features = int(layer.in_features)
    n_rules = int(layer.out_features)
    names = validate_feature_names(feature_names, n_features)
    variant = str(getattr(network, "variant", "unknown"))
    membership = str(getattr(layer, "membership", "gaussian"))
    feature_offset_array = _coordinate_array(
        feature_offsets, n_features, default=0.0, name="feature_offsets"
    )
    feature_scale_array = _coordinate_array(
        feature_scales, n_features, default=1.0, name="feature_scales"
    )
    if np.any(feature_scale_array <= 0):
        raise ValueError("feature_scales must be positive.")

    with torch.no_grad():
        centers = layer.centers.detach().cpu()
        mask = layer.mask.detach().cpu()
        consequent_weights = network.final_layer.weight.detach().cpu()
        consequent_bias = network.final_layer.bias.detach().cpu()
        if variant == "it2":
            lower_widths, upper_widths = layer.widths()
            lower_widths = lower_widths.detach().cpu()
            upper_widths = upper_widths.detach().cpu()
            widths = None
        else:
            widths = layer.widths().detach().cpu()
            lower_widths = None
            upper_widths = None

    n_outputs = int(consequent_weights.shape[0])
    output_offset_array = _coordinate_array(
        output_offsets, n_outputs, default=0.0, name="output_offsets"
    )
    output_scale_array = _coordinate_array(
        output_scales, n_outputs, default=1.0, name="output_scales"
    )
    if np.any(output_scale_array <= 0):
        raise ValueError("output_scales must be positive.")
    if class_labels is None:
        output_names = ["output"] if n_outputs == 1 else [
            f"output_{index}" for index in range(n_outputs)
        ]
    else:
        if len(class_labels) != n_outputs:
            raise ValueError(f"class_labels must contain exactly {n_outputs} labels.")
        output_names = [_label_name(label) for label in class_labels]

    bias_record = {
        output_name: float(
            consequent_bias[output_index].item() * output_scale_array[output_index]
            + output_offset_array[output_index]
        )
        for output_index, output_name in enumerate(output_names)
    }
    input_space = (
        "original"
        if feature_offsets is not None or feature_scales is not None
        else "network"
    )
    output_space = (
        "original"
        if output_offsets is not None or output_scales is not None
        else "network"
    )
    rules: list[dict[str, Any]] = []
    for rule_index in range(n_rules):
        antecedents: list[dict[str, Any]] = []
        for feature_index, feature_name in enumerate(names):
            mask_value = float(mask[feature_index, rule_index].item())
            if abs(mask_value) < threshold:
                continue
            antecedent: dict[str, Any] = {
                "feature": feature_name,
                "mask": mask_value,
                "membership": membership,
                "centers": (
                    centers[feature_index, rule_index].numpy()
                    * feature_scale_array[feature_index]
                    + feature_offset_array[feature_index]
                ).tolist(),
            }
            if variant == "it2":
                antecedent["lower_widths"] = (
                    lower_widths[feature_index, rule_index].numpy()
                    * feature_scale_array[feature_index]
                ).tolist()
                antecedent["upper_widths"] = (
                    upper_widths[feature_index, rule_index].numpy()
                    * feature_scale_array[feature_index]
                ).tolist()
            elif membership == "sigmoid":
                antecedent["slopes"] = (
                    widths[feature_index, rule_index].numpy()
                    / feature_scale_array[feature_index]
                ).tolist()
            else:
                antecedent["widths"] = (
                    widths[feature_index, rule_index].numpy()
                    * feature_scale_array[feature_index]
                ).tolist()
            antecedents.append(antecedent)

        consequents = {
            output_name: float(
                consequent_weights[output_index, rule_index].item()
                * output_scale_array[output_index]
            )
            for output_index, output_name in enumerate(output_names)
        }
        rules.append(
            {
                "rule_index": rule_index,
                "variant": variant,
                "input_space": input_space,
                "output_space": output_space,
                "antecedents": antecedents,
                "consequents": consequents,
                "bias": dict(bias_record),
            }
        )
    return rules


def _coordinate_array(
    values: Sequence[float] | None,
    expected_length: int,
    *,
    default: float,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.full(expected_length, default, dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _format_values(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def rules_to_text(rules: Sequence[dict[str, Any]]) -> str:
    """Render structured rule records as deterministic plain text."""

    blocks: list[str] = []
    for rule in rules:
        lines = [
            f"Rule {rule['rule_index']} [{rule['variant']}]",
            f"  SPACES input={rule['input_space']}, output={rule['output_space']}",
            "  IF",
        ]
        antecedents = rule["antecedents"]
        if not antecedents:
            lines.append("    (no antecedent passes the mask threshold)")
        for antecedent in antecedents:
            lines.append(
                f"    {antecedent['feature']}: membership={antecedent['membership']}, "
                f"mask={antecedent['mask']:.6g}, "
                f"centers={_format_values(antecedent['centers'])}"
            )
            if "widths" in antecedent:
                lines.append(f"      widths={_format_values(antecedent['widths'])}")
            elif "slopes" in antecedent:
                lines.append(f"      slopes={_format_values(antecedent['slopes'])}")
            else:
                lines.append(
                    f"      lower_widths={_format_values(antecedent['lower_widths'])}"
                )
                lines.append(
                    f"      upper_widths={_format_values(antecedent['upper_widths'])}"
                )
        consequent_text = ", ".join(
            f"{name}={value:.6g}" for name, value in rule["consequents"].items()
        )
        bias_text = ", ".join(
            f"{name}={value:.6g}" for name, value in rule["bias"].items()
        )
        lines.append(f"  THEN {consequent_text}")
        lines.append(f"  BIAS {bias_text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_rules(
    rules: Sequence[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Write rules to JSON or text, selected by the destination suffix."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".json":
        destination.write_text(
            json.dumps(list(rules), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif suffix == ".txt":
        destination.write_text(rules_to_text(rules), encoding="utf-8")
    else:
        raise ValueError("Rule output path must end in .json or .txt.")
    return destination


__all__ = ["extract_rules", "rules_to_text", "write_rules"]
