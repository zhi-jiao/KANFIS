"""Fit IT1-KANFIS on the low-dimensional Diabetes regression dataset."""

import argparse
import math
from pathlib import Path

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from kanfis import KANFISRegressor


def main(output_dir: Path) -> None:
    dataset = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
    )

    model = KANFISRegressor(
        variant="it1",
        n_rules=8,
        n_memberships=3,
        max_epochs=100,
        random_state=42,
    )

    # Hyperparameters follow the scikit-learn get_params/set_params convention.
    model.set_params(learning_rate=1e-3, lambda_entropy=0.01)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    print(f"RMSE: {math.sqrt(mean_squared_error(y_test, prediction)):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, prediction):.4f}")

    rules = model.get_rules(feature_names=dataset.feature_names, threshold=0.05)
    print(f"Extracted {len(rules)} rules")
    model.export_rules(
        output_dir / "example_regression_rules.json",
        feature_names=dataset.feature_names,
        threshold=0.05,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    main(parser.parse_args().output_dir)
