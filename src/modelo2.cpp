#include "utils.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

    constexpr std::size_t kNeighbors = 3;

    int checked_mpi_count(std::size_t count)
    {
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::runtime_error("El tamaño de un mensaje excede el límite de MPI");
        }
        return static_cast<int>(count);
    }

    double euclidean_distance(const std::vector<double> &test_features,
                              std::size_t test_offset,
                              const std::vector<double> &train_features,
                              std::size_t train_offset,
                              std::size_t feature_count)
    {
        double squared_distance = 0.0;
        for (std::size_t feature = 0; feature < feature_count; ++feature)
        {
            const double difference =
                test_features[test_offset + feature] - train_features[train_offset + feature];
            squared_distance += difference * difference;
        }
        return std::sqrt(squared_distance);
    }

    int predict(const std::vector<double> &train_features,
                const std::vector<int> &train_targets,
                const std::vector<double> &test_features,
                std::size_t test_offset,
                std::size_t feature_count)
    {
        // Cada proceso tiene el entrenamiento completo, por lo que puede obtener
        // directamente los vecinos definitivos de cada muestra de prueba local.
        std::vector<std::pair<double, int>> distances;
        distances.reserve(train_targets.size());

        for (std::size_t train_row = 0; train_row < train_targets.size(); ++train_row)
        {
            const std::size_t train_offset = train_row * feature_count;
            distances.emplace_back(
                euclidean_distance(test_features, test_offset, train_features,
                                   train_offset, feature_count),
                train_targets[train_row]);
        }

        // El orden por defecto de pair prioriza la distancia y usa la etiqueta
        // como desempate determinista.
        std::sort(distances.begin(), distances.end());

        const std::size_t neighbor_count = std::min(kNeighbors, distances.size());
        std::unordered_map<int, std::size_t> votes;
        int best_label = distances.front().second;
        std::size_t best_votes = 0;

        // Se cuentan únicamente los k vecinos más cercanos.
        for (std::size_t neighbor = 0; neighbor < neighbor_count; ++neighbor)
        {
            const std::size_t current_votes = ++votes[distances[neighbor].second];
            if (current_votes > best_votes)
            {
                best_votes = current_votes;
                best_label = distances[neighbor].second;
            }
        }

        return best_label;
    }

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int process_count = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    try
    {
        if (process_count <= 0)
        {
            throw std::runtime_error("La cantidad de procesos MPI debe ser positiva");
        }

        // El ejecutable se puede lanzar desde la raíz del proyecto o desde src/.
        std::filesystem::path data_directory{"data"};
        if (rank == 0 && !std::filesystem::exists(data_directory / "train.csv"))
        {
            data_directory = "../data";
        }

        std::vector<double> train_features;
        std::vector<int> train_targets;
        std::vector<double> test_features;
        std::vector<int> test_targets;
        std::size_t feature_count = 0;
        std::size_t train_count = 0;
        std::size_t test_count = 0;

        // 1. Lectura de disco (FUERA del cronómetro)
        if (rank == 0)
        {
            std::size_t train_feature_count = 0;
            std::size_t test_feature_count = 0;
            knn::read_csv((data_directory / "train.csv").string(), train_features,
                          train_targets, train_feature_count);
            knn::read_csv((data_directory / "test.csv").string(), test_features,
                          test_targets, test_feature_count);
            if (train_feature_count != test_feature_count)
            {
                throw std::runtime_error("Train y test tienen distinto número de features");
            }
            feature_count = train_feature_count;
            train_count = train_targets.size();
            test_count = test_targets.size();
        }

        // Sincronización e inicio del cronómetro global
        MPI_Barrier(MPI_COMM_WORLD);
        const double start_time = MPI_Wtime();

        // --- FASE DE COMUNICACIÓN 1 (Distribución) ---
        const double t_comm1_start = MPI_Wtime();

        unsigned long long metadata[3] = {
            static_cast<unsigned long long>(train_count),
            static_cast<unsigned long long>(test_count),
            static_cast<unsigned long long>(feature_count)};
        MPI_Bcast(metadata, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        train_count = static_cast<std::size_t>(metadata[0]);
        test_count = static_cast<std::size_t>(metadata[1]);
        feature_count = static_cast<std::size_t>(metadata[2]);

        if (feature_count == 0 || train_count == 0 || test_count == 0)
        {
            throw std::runtime_error("Los datasets no pueden estar vacíos");
        }

        if (rank != 0)
        {
            train_features.resize(train_count * feature_count);
            train_targets.resize(train_count);
        }
        MPI_Bcast(train_features.data(), checked_mpi_count(train_features.size()),
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(train_targets.data(), checked_mpi_count(train_targets.size()),
                  MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> row_counts(process_count);
        std::vector<int> row_displacements(process_count);
        std::vector<int> feature_counts(process_count);
        std::vector<int> feature_displacements(process_count);
        const std::size_t base_rows = test_count / static_cast<std::size_t>(process_count);
        const std::size_t remainder = test_count % static_cast<std::size_t>(process_count);

        std::size_t row_displacement = 0;
        std::size_t feature_displacement = 0;
        for (int process = 0; process < process_count; ++process)
        {
            const std::size_t rows =
                base_rows + (static_cast<std::size_t>(process) < remainder ? 1 : 0);
            row_counts[process] = checked_mpi_count(rows);
            row_displacements[process] = checked_mpi_count(row_displacement);
            feature_counts[process] = checked_mpi_count(rows * feature_count);
            feature_displacements[process] = checked_mpi_count(feature_displacement);
            row_displacement += rows;
            feature_displacement += rows * feature_count;
        }

        const std::size_t local_test_count = static_cast<std::size_t>(row_counts[rank]);
        std::vector<double> local_test_features(local_test_count * feature_count);

        MPI_Scatterv(rank == 0 ? test_features.data() : nullptr, feature_counts.data(),
                     feature_displacements.data(), MPI_DOUBLE,
                     local_test_features.data(),
                     checked_mpi_count(local_test_features.size()), MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);

        const double t_comm1 = MPI_Wtime() - t_comm1_start;

        // --- FASE DE CÓMPUTO PARALELO ---
        const double t_comp_start = MPI_Wtime();
        std::vector<int> local_predictions(local_test_count);
        for (std::size_t test_row = 0; test_row < local_test_count; ++test_row)
        {
            local_predictions[test_row] =
                predict(train_features, train_targets, local_test_features,
                        test_row * feature_count, feature_count);
        }
        const double t_comp = MPI_Wtime() - t_comp_start;

        // --- FASE DE COMUNICACIÓN 2 (Ensamblaje directo) ---
        const double t_comm2_start = MPI_Wtime();
        std::vector<int> predictions;
        if (rank == 0)
        {
            predictions.resize(test_count);
        }
        MPI_Gatherv(local_predictions.data(), checked_mpi_count(local_predictions.size()),
                    MPI_INT, rank == 0 ? predictions.data() : nullptr, row_counts.data(),
                    row_displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);
        const double t_comm2 = MPI_Wtime() - t_comm2_start;

        // Fin del cronómetro global (ANTES de escribir en disco)
        const double elapsed = MPI_Wtime() - start_time;

        if (rank == 0)
        {
            const std::filesystem::path predictions_path = data_directory / "pred_m2.csv";
            std::ofstream output(predictions_path);
            if (!output)
            {
                throw std::runtime_error("No se pudo crear el archivo: " + predictions_path.string());
            }
            for (const int prediction : predictions)
            {
                output << prediction << '\n';
            }

            // Imprime: Tiempo_Total,Tiempo_Comp,Tiempo_Comm
            std::cout << std::fixed << std::setprecision(9)
                      << elapsed << "," << t_comp << "," << (t_comm1 + t_comm2) << '\n';
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Rank " << rank << " - Error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
