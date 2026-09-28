"""Scikit-learn-style estimators for KANFIS."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset

from ._validation import (
    infer_feature_names,
    named_feature_columns,
    validate_estimator_hyperparameters,
    validate_feature_order,
    validate_feature_names,
    validate_features,
    validate_target,
    validate_validation_pair,
)
from .networks import build_network
from .rules import extract_rules, write_rules
from .trainer import KANFISTrainer


class _BaseKANFIS(BaseEstimator):
    """Shared fitting and inference machinery for public KANFIS estimators."""

    _task: str

    def __init__(
        self,
        variant: str = "it1",
        n_rules: int = 8,
        n_memberships: int = 3,
        n_layers: int = 1,
        membership: str = "gaussian",
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        validation_fraction: float = 0.2,
        patience: int = 20,
        min_delta: float = 1e-4,
        lambda_entropy: float = 0.0,
        lambda_distinct: float = 0.0,
        weight_decay: float = 1e-2,
        random_state: int = 42,
        device: str = "auto",
        verbose: int = 1,
    ) -> None:
        self.variant = variant
        self.n_rules = n_rules
        self.n_memberships = n_memberships
        self.n_layers = n_layers
        self.membership = membership
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.lambda_entropy = lambda_entropy
        self.lambda_distinct = lambda_distinct
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _can_stratify(self, encoded_target: np.ndarray) -> bool:
        counts = np.bincount(encoded_target)
        n_validation = int(np.ceil(len(encoded_target) * self.validation_fraction))
        n_training = len(encoded_target) - n_validation
        return (
            counts.size > 1
            and counts.min() >= 2
            and n_validation >= counts.size
            and n_training >= counts.size
        )

    def _split_data(
        self,
        features: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stratify = None
        if self._task == "classification" and self._can_stratify(target):
            stratify = target
        return train_test_split(
            features,
            target,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            shuffle=True,
            stratify=stratify,
        )

    def _make_loaders(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[DataLoader, DataLoader]:
        generator = torch.Generator().manual_seed(int(self.random_state))
        x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
        x_val_tensor = torch.as_tensor(x_val, dtype=torch.float32)
        if self._task == "classification":
            y_train_tensor = torch.as_tensor(y_train, dtype=torch.long)
            y_val_tensor = torch.as_tensor(y_val, dtype=torch.long)
        else:
            y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)
            y_val_tensor = torch.as_tensor(y_val, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(x_train_tensor, y_train_tensor),
            batch_size=min(self.batch_size, len(x_train_tensor)),
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(
            TensorDataset(x_val_tensor, y_val_tensor),
            batch_size=min(self.batch_size, len(x_val_tensor)),
            shuffle=False,
        )
        return train_loader, val_loader

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> "_BaseKANFIS":
        """Fit the estimator and return itself."""

        validate_estimator_hyperparameters(self)
        validate_validation_pair(X_val, y_val)
        features = validate_features(X, min_samples=2)
        target = validate_target(y, task=self._task, n_samples=len(features))
        self.n_features_in_ = features.shape[1]
        self._has_named_features_ = named_feature_columns(X) is not None
        self.feature_names_in_ = np.asarray(
            infer_feature_names(X, self.n_features_in_), dtype=object
        )

        if self._task == "classification":
            self.label_encoder_ = LabelEncoder().fit(target)
            self.classes_ = self.label_encoder_.classes_
            if len(self.classes_) < 2:
                raise ValueError("Classification requires at least two classes.")
            encoded_target = self.label_encoder_.transform(target)
            output_features = len(self.classes_)
        else:
            encoded_target = target
            output_features = 1

        if X_val is None:
            x_train_raw, x_val_raw, y_train_raw, y_val_raw = self._split_data(
                features, encoded_target
            )
        else:
            x_train_raw, y_train_raw = features, encoded_target
            if self._has_named_features_:
                validate_feature_order(X_val, self.feature_names_in_)
            x_val_raw = validate_features(X_val, min_samples=1)
            if x_val_raw.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X_val must contain {self.n_features_in_} features; "
                    f"received {x_val_raw.shape[1]}."
                )
            y_val_checked = validate_target(
                y_val,
                task=self._task,
                n_samples=len(x_val_raw),
                min_samples=1,
            )
            if self._task == "classification":
                try:
                    y_val_raw = self.label_encoder_.transform(y_val_checked)
                except ValueError as error:
                    raise ValueError(
                        "y_val contains a class that is absent from the training target."
                    ) from error
            else:
                y_val_raw = y_val_checked

        self.x_scaler_ = StandardScaler().fit(x_train_raw)
        x_train_scaled = self.x_scaler_.transform(x_train_raw).astype(np.float32)
        x_val_scaled = self.x_scaler_.transform(x_val_raw).astype(np.float32)

        if self._task == "regression":
            self.y_scaler_ = StandardScaler().fit(np.asarray(y_train_raw).reshape(-1, 1))
            y_train_prepared = self.y_scaler_.transform(
                np.asarray(y_train_raw).reshape(-1, 1)
            ).astype(np.float32)
            y_val_prepared = self.y_scaler_.transform(
                np.asarray(y_val_raw).reshape(-1, 1)
            ).astype(np.float32)
        else:
            y_train_prepared = np.asarray(y_train_raw, dtype=np.int64)
            y_val_prepared = np.asarray(y_val_raw, dtype=np.int64)

        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))
        self.network_ = build_network(
            variant=self.variant,
            in_features=self.n_features_in_,
            n_rules=self.n_rules,
            out_features=output_features,
            n_layers=self.n_layers,
            n_memberships=self.n_memberships,
            membership=self.membership,
        )
        self.trainer_ = KANFISTrainer(
            self.network_,
            task=self._task,
            learning_rate=self.learning_rate,
            lambda_entropy=self.lambda_entropy,
            lambda_distinct=self.lambda_distinct,
            weight_decay=self.weight_decay,
            device=self.device,
            verbose=self.verbose,
        )
        train_loader, val_loader = self._make_loaders(
            x_train_scaled,
            y_train_prepared,
            x_val_scaled,
            y_val_prepared,
        )
        self.history_ = self.trainer_.train(
            train_loader,
            val_loader,
            max_epochs=self.max_epochs,
            patience=self.patience,
            min_delta=self.min_delta,
        )
        self.device_ = self.trainer_.device
        return self

    def _predict_raw(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["network_", "x_scaler_", "history_"])
        if self._has_named_features_:
            validate_feature_order(X, self.feature_names_in_)
        features = validate_features(X, min_samples=1)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must contain {self.n_features_in_} features; "
                f"received {features.shape[1]}."
            )
        scaled = self.x_scaler_.transform(features).astype(np.float32)
        tensor = torch.as_tensor(scaled, dtype=torch.float32, device=self.device_)
        self.network_.eval()
        with torch.no_grad():
            prediction, _ = self.network_(tensor)
        return prediction.detach().cpu().numpy()

    def get_rules(
        self,
        feature_names: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return learned first-layer rule parameters."""

        check_is_fitted(self, attributes=["network_", "feature_names_in_"])
        names = validate_feature_names(
            self.feature_names_in_ if feature_names is None else feature_names,
            self.n_features_in_,
        )
        labels = self.classes_ if self._task == "classification" else None
        output_offsets = self.y_scaler_.mean_ if self._task == "regression" else None
        output_scales = self.y_scaler_.scale_ if self._task == "regression" else None
        return extract_rules(
            self.network_,
            feature_names=names,
            class_labels=labels,
            threshold=threshold,
            feature_offsets=self.x_scaler_.mean_,
            feature_scales=self.x_scaler_.scale_,
            output_offsets=output_offsets,
            output_scales=output_scales,
        )

    def export_rules(
        self,
        path: str | Path,
        feature_names: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> Path:
        """Export learned rules to a `.json` or `.txt` file."""

        rules = self.get_rules(feature_names=feature_names, threshold=threshold)
        return write_rules(rules, path)


class KANFISRegressor(RegressorMixin, _BaseKANFIS):
    """Scikit-learn-style single-output KANFIS regressor."""

    _task = "regression"

    def predict(self, X: Any) -> np.ndarray:
        raw_prediction = self._predict_raw(X)
        return self.y_scaler_.inverse_transform(raw_prediction).reshape(-1)


class KANFISClassifier(ClassifierMixin, _BaseKANFIS):
    """Scikit-learn-style binary or multiclass KANFIS classifier."""

    _task = "classification"

    def predict_proba(self, X: Any) -> np.ndarray:
        logits = torch.as_tensor(self._predict_raw(X))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X: Any) -> np.ndarray:
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder_.inverse_transform(class_indices)


__all__ = ["KANFISClassifier", "KANFISRegressor"]
