"""Fit IT2-KANFIS on the four-feature Iris classification dataset."""

import argparse
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from kanfis import KANFISClassifier


def main(output_dir: Path) -> None:
    dataset = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target_names[dataset.target],
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    model = KANFISClassifier(
        variant="it2",
        n_rules=8,
        n_memberships=3,
        learning_rate=1e-3,
        max_epochs=100,
        random_state=42,
    )
    model.fit(x_train, y_train)

    labels = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    print(f"Accuracy: {accuracy_score(y_test, labels):.4f}")
    print(f"Probability shape: {probabilities.shape}")

    rules = model.get_rules(feature_names=dataset.feature_names, threshold=0.05)
    print(rules[0])
    model.export_rules(
        output_dir / "example_classification_rules.txt",
        feature_names=dataset.feature_names,
        threshold=0.05,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    main(parser.parse_args().output_dir)
