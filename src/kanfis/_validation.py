"""Validation helpers shared by KANFIS estimators and rule exporters."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np
from sklearn.utils.validation import check_array


VALID_VARIANTS = {"it1", "it2"}
VALID_MEMBERSHIPS = {"gaussian", "sigmoid", "triangle"}


def _finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and np.isfinite(float(value))
    )


def validate_estimator_hyperparameters(estimator: Any) -> None:
    """Validate public constructor values without mutating the estimator."""

    if not isinstance(estimator.variant, str) or estimator.variant.lower() not in VALID_VARIANTS:
        raise ValueError("variant must be either 'it1' or 'it2'.")
    if (
        not isinstance(estimator.membership, str)
        or estimator.membership.lower() not in VALID_MEMBERSHIPS
    ):
        accepted = ", ".join(sorted(VALID_MEMBERSHIPS))
        raise ValueError(f"membership must be one of: {accepted}.")
    if estimator.variant.lower() == "it2" and estimator.membership.lower() != "gaussian":
        raise ValueError("IT2-KANFIS supports Gaussian membership only.")

    for name in ("n_rules", "n_memberships", "n_layers", "batch_size", "max_epochs", "patience"):
        value = getattr(estimator, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if (
        isinstance(estimator.random_state, bool)
        or not isinstance(estimator.random_state, (int, np.integer))
    ):
        raise ValueError("random_state must be an integer.")
    if (
        isinstance(estimator.verbose, bool)
        or not isinstance(estimator.verbose, (int, np.integer))
        or estimator.verbose < 0
    ):
        raise ValueError("verbose must be a non-negative integer.")

    positive = {"learning_rate": estimator.learning_rate}
    non_negative = {
        "min_delta": estimator.min_delta,
        "lambda_entropy": estimator.lambda_entropy,
        "lambda_distinct": estimator.lambda_distinct,
        "weight_decay": estimator.weight_decay,
    }
    for name, value in positive.items():
        if not _finite_real(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
    for name, value in non_negative.items():
        if not _finite_real(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number.")
    if (
        not _finite_real(estimator.validation_fraction)
        or not 0 < estimator.validation_fraction < 1
    ):
        raise ValueError("validation_fraction must be between 0 and 1.")
    if not isinstance(estimator.device, str):
        raise ValueError("device must be a string such as 'auto', 'cpu', or 'cuda'.")


def validate_validation_pair(X_val: Any, y_val: Any) -> None:
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must be supplied together.")


def validate_features(X: Any, *, min_samples: int = 1) -> np.ndarray:
    """Return a finite two-dimensional float32 feature array."""

    return check_array(
        X,
        accept_sparse=False,
        dtype=np.float32,
        ensure_2d=True,
        ensure_min_samples=min_samples,
        ensure_min_features=1,
    )


def validate_target(
    y: Any,
    *,
    task: str,
    n_samples: int,
    min_samples: int = 2,
) -> np.ndarray:
    """Validate a single-output target while preserving class label dtype."""

    target = np.asarray(y)
    if target.ndim != 1:
        raise ValueError("y must be a one-dimensional target array.")
    if target.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    if target.shape[0] < min_samples:
        raise ValueError(f"At least {min_samples} target value(s) are required.")

    if task == "regression":
        try:
            numeric_target = target.astype(np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("Regression targets must be numeric.") from error
        if not np.isfinite(numeric_target).all():
            raise ValueError("Regression targets must contain only finite values.")
        return numeric_target

    if target.dtype.kind in {"f", "c"} and not np.isfinite(target).all():
        raise ValueError("Classification targets must not contain NaN or infinity.")
    if target.dtype.kind == "O" and any(value is None for value in target):
        raise ValueError("Classification targets must not contain None.")
    return target


def infer_feature_names(X: Any, n_features: int) -> list[str]:
    """Use string DataFrame columns when available, otherwise x0, x1, ..."""

    names = named_feature_columns(X)
    if names is not None and len(names) == n_features:
        return names
    return [f"x{index}" for index in range(n_features)]


def named_feature_columns(X: Any) -> list[str] | None:
    """Return string column labels from a DataFrame-like input, if present."""

    columns = getattr(X, "columns", None)
    if columns is None or not all(isinstance(column, str) for column in columns):
        return None
    return list(columns)


def validate_feature_order(X: Any, expected_names: Sequence[str]) -> None:
    """Reject reordered named columns while continuing to accept arrays."""

    actual_names = named_feature_columns(X)
    if actual_names is not None and actual_names != list(expected_names):
        raise ValueError(
            "Named feature columns must match the fitted columns in the same order."
        )


def validate_feature_names(
    feature_names: Sequence[str] | None,
    n_features: int,
) -> list[str]:
    if feature_names is None:
        return [f"x{index}" for index in range(n_features)]
    names = list(feature_names)
    if len(names) != n_features:
        raise ValueError(f"feature_names must contain exactly {n_features} names.")
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Every feature name must be a non-empty string.")
    return names
