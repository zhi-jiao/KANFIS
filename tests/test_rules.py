import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

from kanfis import KANFISClassifier, KANFISRegressor
from kanfis.networks import IT1KANFISNetwork, IT2KANFISNetwork
from kanfis.rules import extract_rules, write_rules


def test_it1_rule_schema_contains_widths_and_stable_indices():
    network = IT1KANFISNetwork(2, 2, 1, n_memberships=2)
    network.layers[0].mask.data[:, 0] = 0.9
    network.layers[0].mask.data[:, 1] = 0.1

    rules = extract_rules(network, feature_names=["a", "b"], threshold=0.5)

    assert [rule["rule_index"] for rule in rules] == [0, 1]
    assert len(rules[0]["antecedents"]) == 2
    assert rules[1]["antecedents"] == []
    assert "widths" in rules[0]["antecedents"][0]
    assert "lower_widths" not in rules[0]["antecedents"][0]


def test_it2_rules_contain_ordered_interval_widths():
    network = IT2KANFISNetwork(2, 1, 1, n_memberships=2)

    rules = extract_rules(network, feature_names=["a", "b"])
    antecedent = rules[0]["antecedents"][0]

    assert "lower_widths" in antecedent
    assert "upper_widths" in antecedent
    assert np.all(np.asarray(antecedent["upper_widths"]) > antecedent["lower_widths"])


def test_rule_parameters_can_be_reported_in_original_units():
    network = IT1KANFISNetwork(2, 1, 1, n_memberships=2)
    network.layers[0].centers.data.zero_()
    network.final_layer.weight.data.fill_(2.0)
    network.final_layer.bias.data.fill_(3.0)

    rules = extract_rules(
        network,
        feature_names=["a", "b"],
        feature_offsets=[10.0, 20.0],
        feature_scales=[2.0, 4.0],
        output_offsets=[100.0],
        output_scales=[5.0],
    )

    assert rules[0]["input_space"] == "original"
    assert rules[0]["output_space"] == "original"
    assert rules[0]["antecedents"][0]["centers"] == [10.0, 10.0]
    assert rules[0]["consequents"]["output"] == 10.0
    assert rules[0]["bias"]["output"] == 115.0


def test_classifier_rule_consequents_use_original_labels():
    x, y = make_classification(
        n_samples=30,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        random_state=3,
    )
    labels = np.where(y == 0, "negative", "positive")
    model = KANFISClassifier(
        n_rules=2,
        max_epochs=1,
        batch_size=10,
        device="cpu",
        verbose=0,
    ).fit(x, labels)

    rules = model.get_rules(feature_names=["x", "y", "z"])

    assert set(rules[0]["consequents"]) == {"negative", "positive"}


def test_rule_exports_are_json_and_human_readable(tmp_path):
    network = IT1KANFISNetwork(2, 1, 1)
    rules = extract_rules(network, feature_names=["a", "b"])
    json_path = write_rules(rules, tmp_path / "rules.json")
    text_path = write_rules(rules, tmp_path / "rules.txt")

    assert json.loads(json_path.read_text(encoding="utf-8")) == rules
    assert "Rule 0 [it1]" in text_path.read_text(encoding="utf-8")


def test_rule_methods_require_fitted_estimator(tmp_path):
    model = KANFISRegressor()

    with pytest.raises(NotFittedError):
        model.get_rules()
    with pytest.raises(NotFittedError):
        model.export_rules(tmp_path / "rules.json")


def test_rule_writer_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ValueError, match=".json or .txt"):
        write_rules([], tmp_path / "rules.csv")


def test_rule_extraction_rejects_deep_network_consequent_misattribution():
    network = IT1KANFISNetwork(2, 2, 1, n_layers=2)

    with pytest.raises(ValueError, match="n_layers=1"):
        extract_rules(network)
