from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.axes import Axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ("Secuencial", "Modelo1", "Modelo2")
PARALLEL_MODELS = ("Modelo1", "Modelo2")
MODEL_COLORS = {
    "Secuencial": "tab:blue",
    "Modelo1": "tab:orange",
    "Modelo2": "tab:green",
}
DATASET_LINESTYLES = ("-", "--", ":")
PROCESS_TICKS = (1, 2, 4, 8, 16)


def load_benchmark(path: Path) -> pd.DataFrame:
    """Carga y valida el CSV producido por run_benchmark.sh."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el benchmark: {path}")

    required_columns = {
        "Modelo",
        "Tamano_Dataset",
        "Procesos",
        "Tiempo_Segundos",
    }
    results = pd.read_csv(path)
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")

    results = results.copy()
    results["Tamano_Dataset"] = pd.to_numeric(results["Tamano_Dataset"])
    results["Procesos"] = pd.to_numeric(results["Procesos"])
    results["Tiempo_Segundos"] = pd.to_numeric(results["Tiempo_Segundos"])
    if (results["Tiempo_Segundos"] <= 0).any():
        raise ValueError("Todos los tiempos deben ser mayores que cero")
    return results


def calculate_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Calcula speedup y eficiencia usando el tiempo secuencial de cada N."""
    sequential = results[results["Modelo"] == "Secuencial"]
    baseline = sequential.set_index("Tamano_Dataset")["Tiempo_Segundos"]

    metrics = results[results["Modelo"].isin(PARALLEL_MODELS)].copy()
    metrics["Tiempo_Secuencial"] = metrics["Tamano_Dataset"].map(baseline)
    if metrics["Tiempo_Secuencial"].isna().any():
        raise ValueError("Falta el tiempo secuencial para uno de los datasets")

    metrics["Speedup"] = (
        metrics["Tiempo_Secuencial"] / metrics["Tiempo_Segundos"]
    )
    metrics["Eficiencia"] = metrics["Speedup"] / metrics["Procesos"]
    return metrics


def configure_process_axis(axis: Axes, ylabel: str) -> None:
    """Configura el eje de procesos común a las tres gráficas."""
    axis.set_xscale("log", base=2)
    axis.set_xticks(PROCESS_TICKS)
    axis.set_xticklabels([str(processes) for processes in PROCESS_TICKS])
    axis.set_xlabel("Procesos")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.25)


def plot_execution_time(
    axis: Axes, results: pd.DataFrame, datasets: list[int]
) -> None:
    """Grafica tiempos para los tres modelos y todos los tamaños."""
    for dataset_index, dataset_size in enumerate(datasets):
        linestyle = DATASET_LINESTYLES[dataset_index % len(DATASET_LINESTYLES)]
        dataset_results = results[results["Tamano_Dataset"] == dataset_size]
        for model in MODELS:
            model_results = dataset_results[dataset_results["Modelo"] == model]
            if model_results.empty:
                continue
            model_results = model_results.sort_values("Procesos")
            axis.plot(
                model_results["Procesos"],
                model_results["Tiempo_Segundos"],
                color=MODEL_COLORS[model],
                linestyle=linestyle,
                marker="o",
                label=f"{model}, N={dataset_size}",
            )

    configure_process_axis(axis, "Tiempo (s)")
    axis.set_title("Tiempo de ejecución")
    axis.legend(fontsize="small")


def plot_speedup(
    axis: Axes, metrics: pd.DataFrame, datasets: list[int]
) -> None:
    """Grafica speedup experimental e ideal."""
    max_processes = max(PROCESS_TICKS)
    axis.plot(
        PROCESS_TICKS,
        PROCESS_TICKS,
        color="black",
        linestyle="--",
        label="Ideal",
    )

    for dataset_index, dataset_size in enumerate(datasets):
        linestyle = DATASET_LINESTYLES[dataset_index % len(DATASET_LINESTYLES)]
        dataset_metrics = metrics[metrics["Tamano_Dataset"] == dataset_size]
        for model in PARALLEL_MODELS:
            model_metrics = dataset_metrics[dataset_metrics["Modelo"] == model]
            if model_metrics.empty:
                continue
            model_metrics = model_metrics.sort_values("Procesos")
            axis.plot(
                model_metrics["Procesos"],
                model_metrics["Speedup"],
                color=MODEL_COLORS[model],
                linestyle=linestyle,
                marker="o",
                label=f"{model}, N={dataset_size}",
            )

    configure_process_axis(axis, "Speedup")
    axis.set_xlim(1, max_processes)
    axis.set_title("Speedup")
    axis.legend(fontsize="small")


def plot_efficiency(
    axis: Axes, metrics: pd.DataFrame, datasets: list[int]
) -> None:
    """Grafica eficiencia experimental y la referencia ideal del 100 %."""
    axis.axhline(1.0, color="black", linestyle="--", label="Ideal")

    for dataset_index, dataset_size in enumerate(datasets):
        linestyle = DATASET_LINESTYLES[dataset_index % len(DATASET_LINESTYLES)]
        dataset_metrics = metrics[metrics["Tamano_Dataset"] == dataset_size]
        for model in PARALLEL_MODELS:
            model_metrics = dataset_metrics[dataset_metrics["Modelo"] == model]
            if model_metrics.empty:
                continue
            model_metrics = model_metrics.sort_values("Procesos")
            axis.plot(
                model_metrics["Procesos"],
                model_metrics["Eficiencia"],
                color=MODEL_COLORS[model],
                linestyle=linestyle,
                marker="o",
                label=f"{model}, N={dataset_size}",
            )

    configure_process_axis(axis, "Eficiencia")
    axis.set_ylim(bottom=0)
    axis.set_title("Eficiencia")
    axis.legend(fontsize="small")


def main() -> None:
    """Genera y guarda la figura consolidada de rendimiento."""
    project_root = Path(__file__).resolve().parent.parent
    results_path = project_root / "data" / "resultados_benchmark.csv"
    output_path = project_root / "data" / "analisis_rendimiento.png"

    results = load_benchmark(results_path)
    metrics = calculate_metrics(results)
    datasets = sorted(results["Tamano_Dataset"].unique().tolist())

    figure, axes = plt.subplots(1, 3, figsize=(21, 6))
    plot_execution_time(axes[0], results, datasets)
    plot_speedup(axes[1], metrics, datasets)
    plot_efficiency(axes[2], metrics, datasets)
    figure.suptitle("Análisis de rendimiento KNN con MPI")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Gráfico guardado en: {output_path}")


if __name__ == "__main__":
    main()
