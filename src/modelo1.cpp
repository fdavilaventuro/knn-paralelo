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

    void local_candidates(const std::vector<double> &train_features,
                          const std::vector<int> &train_targets,
                          const std::vector<double> &test_features,
                          std::size_t feature_count,
                          std::vector<double> &candidate_distances,
                          std::vector<int> &candidate_labels)
    {
        const std::size_t test_count = test_features.size() / feature_count;
        const std::size_t local_train_count = train_targets.size();
        const double infinity = std::numeric_limits<double>::infinity();

        candidate_distances.assign(test_count * kNeighbors, infinity);
        candidate_labels.assign(test_count * kNeighbors, 0);

        for (std::size_t test_row = 0; test_row < test_count; ++test_row)
        {
            std::vector<std::pair<double, int>> distances;
            distances.reserve(local_train_count);
            const std::size_t test_offset = test_row * feature_count;

            for (std::size_t train_row = 0; train_row < local_train_count; ++train_row)
            {
                const std::size_t train_offset = train_row * feature_count;
                distances.emplace_back(
                    euclidean_distance(test_features, test_offset, train_features,
                                       train_offset, feature_count),
                    train_targets[train_row]);
            }

            std::sort(distances.begin(), distances.end());
            const std::size_t count = std::min(kNeighbors, distances.size());
            for (std::size_t neighbor = 0; neighbor < count; ++neighbor)
            {
                const std::size_t candidate_offset = test_row * kNeighbors + neighbor;
                candidate_distances[candidate_offset] = distances[neighbor].first;
                candidate_labels[candidate_offset] = distances[neighbor].second;
            }
        }
    }

    int majority_vote(std::vector<std::pair<double, int>> &candidates)
    {
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                           [](const std::pair<double, int> &candidate)
                           {
                               return !std::isfinite(candidate.first);
                           }),
            candidates.end());
        if (candidates.empty())
        {
            throw std::runtime_error("No hay candidatos para calcular una predicción");
        }

        std::sort(candidates.begin(), candidates.end());
        const std::size_t count = std::min(kNeighbors, candidates.size());
        std::unordered_map<int, std::size_t> votes;
        int best_label = candidates.front().second;
        std::size_t best_votes = 0;

        for (std::size_t neighbor = 0; neighbor < count; ++neighbor)
        {
            const std::size_t current_votes = ++votes[candidates[neighbor].second];
            if (current_votes > best_votes)
            {
                best_votes = current_votes;
                best_label = candidates[neighbor].second;
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
            test_features.resize(test_count * feature_count);
        }
        MPI_Bcast(test_features.data(), checked_mpi_count(test_features.size()),
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<int> row_counts(process_count);
        std::vector<int> row_displacements(process_count);
        std::vector<int> feature_counts(process_count);
        std::vector<int> feature_displacements(process_count);
        const std::size_t base_rows = train_count / static_cast<std::size_t>(process_count);
        const std::size_t remainder = train_count % static_cast<std::size_t>(process_count);

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

        const std::size_t local_train_count = static_cast<std::size_t>(row_counts[rank]);
        std::vector<double> local_train_features(local_train_count * feature_count);
        std::vector<int> local_train_targets(local_train_count);

        MPI_Scatterv(rank == 0 ? train_features.data() : nullptr, feature_counts.data(),
                     feature_displacements.data(), MPI_DOUBLE,
                     local_train_features.data(),
                     checked_mpi_count(local_train_features.size()), MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? train_targets.data() : nullptr, row_counts.data(),
                     row_displacements.data(), MPI_INT, local_train_targets.data(),
                     checked_mpi_count(local_train_targets.size()), MPI_INT, 0,
                     MPI_COMM_WORLD);

        const double t_comm1 = MPI_Wtime() - t_comm1_start;

        // --- FASE DE CÓMPUTO PARALELO ---
        const double t_comp_start = MPI_Wtime();
        std::vector<double> local_distances;
        std::vector<int> local_labels;
        local_candidates(local_train_features, local_train_targets, test_features,
                         feature_count, local_distances, local_labels);
        const double t_comp = MPI_Wtime() - t_comp_start;

        // --- FASE DE COMUNICACIÓN 2 Y REDUCCIÓN ---
        const double t_comm2_start = MPI_Wtime();
        const std::size_t candidates_per_process = test_count * kNeighbors;
        std::vector<double> all_distances;
        std::vector<int> all_labels;
        if (rank == 0)
        {
            all_distances.resize(static_cast<std::size_t>(process_count) * candidates_per_process);
            all_labels.resize(static_cast<std::size_t>(process_count) * candidates_per_process);
        }

        MPI_Gather(local_distances.data(), checked_mpi_count(candidates_per_process),
                   MPI_DOUBLE, rank == 0 ? all_distances.data() : nullptr,
                   checked_mpi_count(candidates_per_process), MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(local_labels.data(), checked_mpi_count(candidates_per_process),
                   MPI_INT, rank == 0 ? all_labels.data() : nullptr,
                   checked_mpi_count(candidates_per_process), MPI_INT, 0,
                   MPI_COMM_WORLD);

        std::vector<int> predictions;
        if (rank == 0)
        {
            predictions.reserve(test_count);
            for (std::size_t test_row = 0; test_row < test_count; ++test_row)
            {
                std::vector<std::pair<double, int>> candidates;
                candidates.reserve(static_cast<std::size_t>(process_count) * kNeighbors);
                for (int process = 0; process < process_count; ++process)
                {
                    const std::size_t process_offset =
                        static_cast<std::size_t>(process) * candidates_per_process;
                    for (std::size_t neighbor = 0; neighbor < kNeighbors; ++neighbor)
                    {
                        const std::size_t offset =
                            process_offset + test_row * kNeighbors + neighbor;
                        candidates.emplace_back(all_distances[offset], all_labels[offset]);
                    }
                }
                predictions.push_back(majority_vote(candidates));
            }
        }
        const double t_comm2 = MPI_Wtime() - t_comm2_start;

        // Fin del cronómetro global (ANTES de escribir en disco)
        const double elapsed = MPI_Wtime() - start_time;

        if (rank == 0)
        {
            const std::filesystem::path predictions_path = data_directory / "pred_m1.csv";
            std::ofstream output(predictions_path);
            if (!output)
            {
                throw std::runtime_error("No se pudo crear el archivo: " + predictions_path.string());
            }
            for (const int prediction : predictions)
            {
                output << prediction << '\n';
            }

            // Imprime: Tiempo_Total,Tiempo_Comp,Tiempo_Comm_y_Red
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
