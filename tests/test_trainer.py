import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kanfis import IT1KANFISNetwork, KANFISTrainer
from kanfis.trainer import EarlyStopping


def _regression_loader():
    x = torch.randn(24, 3)
    y = (x[:, :1] * 2.0) - 0.5
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def _classification_loader():
    x = torch.randn(24, 3)
    y = (x[:, 0] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def test_regression_trainer_returns_complete_history():
    network = IT1KANFISNetwork(3, 2, 1)
    trainer = KANFISTrainer(network, task="regression", device="cpu", verbose=0)

    history = trainer.train(_regression_loader(), max_epochs=2, patience=2)

    assert set(history) == {"train_loss", "val_loss", "epochs_trained", "best_epoch"}
    assert history["epochs_trained"] == len(history["train_loss"])
    assert len(history["val_loss"]) == history["epochs_trained"]


def test_classification_trainer_accepts_integer_targets():
    network = IT1KANFISNetwork(3, 2, 2)
    trainer = KANFISTrainer(network, task="classification", device="cpu", verbose=0)

    history = trainer.train(_classification_loader(), max_epochs=2, patience=2)

    assert history["epochs_trained"] >= 1
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))


def test_early_stopping_restores_a_deep_copy_of_best_state():
    model = nn.Linear(2, 1)
    stopper = EarlyStopping(patience=1, min_delta=0.0)
    stopper.step(model, 1.0)
    expected = copy.deepcopy(model.state_dict())

    with torch.no_grad():
        model.weight.add_(10.0)
    should_stop = stopper.step(model, 2.0)

    assert should_stop
    assert torch.equal(model.weight, expected["weight"])


def test_one_rule_distinctness_loss_is_finite_zero():
    network = IT1KANFISNetwork(3, 1, 1)
    trainer = KANFISTrainer(network, task="regression", device="cpu", verbose=0)

    loss = trainer._distinctness_loss(torch.ones(8, 1))

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
