"""Shared PyTorch training loop for KANFIS networks."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from numbers import Integral, Real
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Integral) and value >= 1


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve an explicit device or choose CUDA when `device='auto'`."""

    if isinstance(device, torch.device):
        resolved = device
    elif device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            resolved = torch.device(device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"Invalid device {device!r}.") from error

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return resolved


class EarlyStopping:
    """Track validation loss and restore an independent best-state snapshot."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4) -> None:
        if not _positive_integer(patience):
            raise ValueError("patience must be a positive integer.")
        if not _finite_number(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a non-negative finite number.")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = math.inf
        self.best_epoch = 0
        self.counter = 0
        self.best_state: dict[str, Any] | None = None

    def step(self, model: nn.Module, loss: float, epoch: int | None = None) -> bool:
        """Update state and return whether training should stop."""

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_epoch = 0 if epoch is None else epoch
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.restore(model)
            return True
        return False

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class KANFISTrainer:
    """Train a KANFIS network for regression or classification."""

    VALID_TASKS = {"regression", "classification"}

    def __init__(
        self,
        network: nn.Module,
        task: str,
        learning_rate: float = 1e-3,
        lambda_entropy: float = 0.0,
        lambda_distinct: float = 0.0,
        weight_decay: float = 1e-2,
        device: str | torch.device = "auto",
        verbose: int = 1,
    ) -> None:
        task = task.lower()
        if task not in self.VALID_TASKS:
            raise ValueError("task must be either 'regression' or 'classification'.")
        if not _finite_number(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number.")
        if any(
            not _finite_number(value) or value < 0
            for value in (lambda_entropy, lambda_distinct, weight_decay)
        ):
            raise ValueError("Regularization values must be non-negative and finite.")
        if (
            isinstance(verbose, bool)
            or not isinstance(verbose, Integral)
            or verbose < 0
        ):
            raise ValueError("verbose must be a non-negative integer.")

        self.network = network
        self.task = task
        self.learning_rate = float(learning_rate)
        self.lambda_entropy = float(lambda_entropy)
        self.lambda_distinct = float(lambda_distinct)
        self.weight_decay = float(weight_decay)
        self.device = resolve_device(device)
        self.verbose = int(verbose)
        self.network.to(self.device)

        mask_parameters: list[nn.Parameter] = []
        base_parameters: list[nn.Parameter] = []
        for name, parameter in self.network.named_parameters():
            if "mask" in name:
                mask_parameters.append(parameter)
            else:
                base_parameters.append(parameter)

        parameter_groups: list[dict[str, Any]] = [
            {
                "params": base_parameters,
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            }
        ]
        if mask_parameters:
            parameter_groups.append(
                {
                    "params": mask_parameters,
                    "lr": self.learning_rate * 10.0,
                    "weight_decay": 0.0,
                }
            )
        self.optimizer = torch.optim.AdamW(parameter_groups)

    def _task_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            return F.cross_entropy(prediction, target.reshape(-1).long())
        regression_target = target.float()
        if regression_target.ndim == 1:
            regression_target = regression_target[:, None]
        return F.mse_loss(prediction, regression_target)

    def _entropy_loss(self, mask: torch.Tensor) -> torch.Tensor:
        probability = torch.abs(mask)
        probability = probability / (probability.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(probability * torch.log(probability + 1e-8)).sum(dim=1)
        return entropy.mean()

    def _distinctness_loss(self, firing_strengths: torch.Tensor | None) -> torch.Tensor:
        if firing_strengths is None or firing_strengths.shape[1] <= 1:
            return torch.zeros((), device=self.device)
        normalized_rules = F.normalize(firing_strengths.transpose(0, 1), p=2, dim=1)
        similarity = normalized_rules @ normalized_rules.transpose(0, 1)
        n_rules = similarity.shape[0]
        off_diagonal = similarity - torch.eye(n_rules, device=similarity.device)
        return torch.square(off_diagonal).sum() / (n_rules * (n_rules - 1))

    def _validation_loss(self, loader: Iterable[Any]) -> float:
        self.network.eval()
        total = 0.0
        samples = 0
        with torch.no_grad():
            for features, target in loader:
                features = features.to(self.device)
                target = target.to(self.device)
                prediction, _ = self.network(features)
                batch_size = int(features.shape[0])
                total += self._task_loss(prediction, target).item() * batch_size
                samples += batch_size
        if samples == 0:
            raise ValueError("Validation loader must contain at least one batch.")
        average = total / samples
        if not math.isfinite(average):
            raise FloatingPointError("Non-finite validation loss.")
        return average

    def train(
        self,
        train_loader: Iterable[Any],
        val_loader: Iterable[Any] | None = None,
        max_epochs: int = 100,
        patience: int = 20,
        min_delta: float = 1e-4,
    ) -> dict[str, list[float] | int]:
        """Run the optimization loop and return serializable loss history."""

        if not _positive_integer(max_epochs):
            raise ValueError("max_epochs must be a positive integer.")
        max_epochs = int(max_epochs)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        warmup_epochs = max(1, int(max_epochs * 0.3))
        history: dict[str, list[float] | int] = {
            "train_loss": [],
            "val_loss": [],
            "epochs_trained": 0,
            "best_epoch": 0,
        }

        for epoch_index in range(max_epochs):
            self.network.train()
            task_loss_sum = 0.0
            samples = 0
            entropy_weight = self.lambda_entropy * min(
                1.0, epoch_index / warmup_epochs
            )

            for features, target in train_loader:
                features = features.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                prediction, firing_strengths = self.network(features)
                task_loss = self._task_loss(prediction, target)
                entropy_loss = entropy_weight * self._entropy_loss(
                    self.network.get_mask()
                )
                distinctness_loss = self.lambda_distinct * self._distinctness_loss(
                    firing_strengths
                )
                loss = task_loss + entropy_loss + distinctness_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch_index + 1}."
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()
                batch_size = int(features.shape[0])
                task_loss_sum += task_loss.item() * batch_size
                samples += batch_size

            if samples == 0:
                raise ValueError("Training loader must contain at least one batch.")
            train_loss = task_loss_sum / samples
            val_loss = (
                self._validation_loss(val_loader)
                if val_loader is not None
                else train_loss
            )
            history["train_loss"].append(float(train_loss))  # type: ignore[union-attr]
            history["val_loss"].append(float(val_loss))  # type: ignore[union-attr]
            history["epochs_trained"] = epoch_index + 1

            should_stop = early_stopping.step(
                self.network, val_loss, epoch=epoch_index + 1
            )
            history["best_epoch"] = early_stopping.best_epoch

            if self.verbose and (
                epoch_index < 5 or (epoch_index + 1) % 10 == 0 or should_stop
            ):
                print(
                    f"Epoch {epoch_index + 1}: train_loss={train_loss:.6f}, "
                    f"val_loss={val_loss:.6f}"
                )
            if should_stop:
                break

        early_stopping.restore(self.network)
        return history


__all__ = ["EarlyStopping", "KANFISTrainer", "resolve_device"]
