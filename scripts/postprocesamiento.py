from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_predictions(path: Path, expected_count: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de predicciones: {path}")

    predictions = np.atleast_1d(np.loadtxt(path, dtype=int))
    if predictions.size != expected_count:
        raise ValueError(
            f"{path} contiene {predictions.size} predicciones; "
            f"se esperaban {expected_count}"
        )
    return predictions


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_directory = project_root / "data"
    test_path = data_directory / "test.csv"

    test_data = np.loadtxt(test_path, delimiter=",")
    images = test_data[:, :64]
    real_labels = test_data[:, -1].astype(int)

    prediction_files = {
        "Secuencial": data_directory / "pred_secuencial.csv",
        "Modelo 1": data_directory / "pred_m1.csv",
        "Modelo 2": data_directory / "pred_m2.csv",
    }
    predictions = {
        name: load_predictions(path, real_labels.size)
        for name, path in prediction_files.items()
    }

    for name, model_predictions in predictions.items():
        accuracy = np.mean(model_predictions == real_labels)
        print(f"{name}: accuracy = {accuracy:.4f}")

    figure, axes = plt.subplots(2, 5, figsize=(12, 5))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index].reshape(8, 8), cmap="gray")
        axis.set_title(
            f"Real: {real_labels[index]}\n"
            f"Pred: {predictions['Modelo 2'][index]}"
        )
        axis.axis("off")

    figure.tight_layout()
    output_path = project_root / "grafico_resultados.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Gráfico guardado en: {output_path}")


if __name__ == "__main__":
    main()
