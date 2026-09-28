import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError

from kanfis import KANFISClassifier, KANFISRegressor


class _NamedArray:
    def __init__(self, values, columns):
        self.values = np.asarray(values)
        self.columns = list(columns)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


def _regression_data():
    return make_regression(
        n_samples=48,
        n_features=3,
        n_informative=3,
        noise=0.1,
        random_state=7,
    )


def _classification_data():
    return make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        random_state=7,
    )


@pytest.mark.parametrize("variant", ["it1", "it2"])
def test_regressor_fits_and_restores_target_scale(variant):
    x, y = _regression_data()
    model = KANFISRegressor(
        variant=variant,
        n_rules=3,
        n_memberships=2,
        max_epochs=2,
        patience=2,
        batch_size=16,
        random_state=11,
        device="cpu",
        verbose=0,
    )

    model.fit(x, y)
    prediction = model.predict(x[:5])
    scaled_prediction = model._predict_raw(x[:5])

    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
    assert np.allclose(
        prediction,
        model.y_scaler_.inverse_transform(scaled_prediction).reshape(-1),
    )
    assert not hasattr(model, "predict_proba")


@pytest.mark.parametrize("variant", ["it1", "it2"])
def test_classifier_preserves_string_labels_and_normalizes_probabilities(variant):
    x, y = _classification_data()
    labels = np.asarray([f"class-{value}" for value in y])
    model = KANFISClassifier(
        variant=variant,
        n_rules=3,
        n_memberships=2,
        max_epochs=2,
        patience=2,
        batch_size=16,
        random_state=11,
        device="cpu",
        verbose=0,
    )

    model.fit(x, labels)
    prediction = model.predict(x[:6])
    probability = model.predict_proba(x[:6])

    assert prediction.shape == (6,)
    assert set(prediction).issubset(set(labels))
    assert probability.shape == (6, 3)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.array_equal(model.classes_, np.unique(labels))


def test_fixed_seed_produces_deterministic_predictions():
    x, y = _regression_data()
    parameters = {
        "n_rules": 2,
        "max_epochs": 2,
        "batch_size": 16,
        "random_state": 19,
        "device": "cpu",
        "verbose": 0,
    }

    first = KANFISRegressor(**parameters).fit(x, y)
    second = KANFISRegressor(**parameters).fit(x, y)

    assert np.allclose(first.predict(x[:8]), second.predict(x[:8]))


def test_sklearn_parameter_interface_updates_next_fit_configuration():
    model = KANFISRegressor(n_rules=4)

    returned = model.set_params(n_rules=6, learning_rate=0.005)

    assert returned is model
    assert model.get_params()["n_rules"] == 6
    assert model.get_params()["learning_rate"] == 0.005


def test_fit_rejects_incomplete_explicit_validation_pair():
    x, y = _regression_data()

    with pytest.raises(ValueError, match="X_val and y_val"):
        KANFISRegressor(max_epochs=1).fit(x, y, X_val=x[:5])


def test_invalid_variant_is_rejected_before_training():
    x, y = _regression_data()

    with pytest.raises(ValueError, match="variant"):
        KANFISRegressor(variant="unknown").fit(x, y)


def test_predict_requires_fit_and_matching_feature_count():
    model = KANFISRegressor(max_epochs=1, verbose=0)
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((2, 3)))

    x, y = _regression_data()
    model.fit(x, y)
    with pytest.raises(ValueError, match="3 features"):
        model.predict(np.zeros((2, 2)))


def test_named_feature_order_is_enforced_during_inference():
    x, y = _regression_data()
    named_x = _NamedArray(x, ["a", "b", "c"])
    model = KANFISRegressor(
        n_rules=2,
        max_epochs=1,
        batch_size=16,
        device="cpu",
        verbose=0,
    ).fit(named_x, y)

    reordered = _NamedArray(x[:3, [1, 0, 2]], ["b", "a", "c"])
    with pytest.raises(ValueError, match="same order"):
        model.predict(reordered)
