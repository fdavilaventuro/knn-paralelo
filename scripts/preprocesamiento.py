import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera datasets de KNN con data augmentation gaussiano."
    )
    parser.add_argument(
        "N",
        type=int,
        help="Número total de muestras deseado (debe ser al menos 1797).",
    )
    args = parser.parse_args()

    digits = load_digits()
    original_features = digits.data
    original_targets = digits.target
    original_count = original_features.shape[0]
    if args.N < original_count:
        parser.error(
            f"N debe ser mayor o igual que las {original_count} muestras originales."
        )

    missing_count = args.N - original_count
    rng = np.random.default_rng(42)
    selected_indices = rng.integers(0, original_count, size=missing_count)
    augmented_features = np.clip(
        original_features[selected_indices]
        + rng.normal(0.0, 0.5, size=(missing_count, original_features.shape[1])),
        0,
        16,
    )
    augmented_targets = original_targets[selected_indices]

    features = np.concatenate((original_features, augmented_features), axis=0)
    targets = np.concatenate((original_targets, augmented_targets), axis=0)

    features_train, features_test, targets_train, targets_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=42,
    )

    data_directory = Path(__file__).resolve().parent.parent / "data"
    data_directory.mkdir(parents=True, exist_ok=True)

    train_rows = np.column_stack((features_train, targets_train))
    test_rows = np.column_stack((features_test, targets_test))

    for output_path, rows in (
        (data_directory / "train.csv", train_rows),
        (data_directory / "test.csv", test_rows),
    ):
        with output_path.open("w", newline="") as output_file:
            csv.writer(output_file).writerows(rows.tolist())

    print(
        f"Preprocesamiento completado correctamente: "
        f"{args.N} muestras ({missing_count} aumentadas)."
    )


if __name__ == "__main__":
    main()
