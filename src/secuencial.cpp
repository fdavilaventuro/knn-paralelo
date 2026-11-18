#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

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
                std::size_t feature_count,
                std::size_t k)
    {
        std::vector<std::pair<double, int>> distances;
        distances.reserve(train_targets.size());

        for (std::size_t train_row = 0; train_row < train_targets.size(); ++train_row)
        {
            const std::size_t train_offset = train_row * feature_count;
            distances.emplace_back(euclidean_distance(test_features, test_offset,
                                                      train_features, train_offset, feature_count),
                                   train_targets[train_row]);
        }

        std::sort(distances.begin(), distances.end());

        const std::size_t neighbor_count = std::min(k, distances.size());
        std::unordered_map<int, std::size_t> votes;
        int best_label = distances.front().second;
        std::size_t best_votes = 0;

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

int main()
{
    try
    {
        std::filesystem::path data_directory{"data"};
        if (!std::filesystem::exists(data_directory / "train.csv"))
        {
            data_directory = "../data";
        }

        const std::filesystem::path train_path = data_directory / "train.csv";
        const std::filesystem::path test_path = data_directory / "test.csv";
        const std::filesystem::path predictions_path =
            data_directory / "pred_secuencial.csv";

        std::vector<double> train_features;
        std::vector<int> train_targets;
        std::size_t train_feature_count = 0;
        knn::read_csv(train_path.string(), train_features, train_targets, train_feature_count);

        std::vector<double> test_features;
        std::vector<int> test_targets;
        std::size_t test_feature_count = 0;
        knn::read_csv(test_path.string(), test_features, test_targets, test_feature_count);

        if (train_feature_count != test_feature_count)
        {
            throw std::runtime_error("Train y test tienen distinto número de features");
        }

        const auto start = std::chrono::steady_clock::now();
        std::vector<int> predictions;
        predictions.reserve(test_targets.size());
        for (std::size_t test_row = 0; test_row < test_targets.size(); ++test_row)
        {
            predictions.push_back(predict(train_features, train_targets,
                                          test_features, test_row * test_feature_count,
                                          train_feature_count, 3));
        }
        const auto end = std::chrono::steady_clock::now();

        std::ofstream output(predictions_path);
        if (!output)
        {
            throw std::runtime_error("No se pudo crear el archivo: " +
                                     predictions_path.string());
        }
        for (const int prediction : predictions)
        {
            output << prediction << '\n';
        }

        const double elapsed = std::chrono::duration<double>(end - start).count();

        // Imprime: Tiempo_Total,Tiempo_Comp,Tiempo_Comm
        std::cout << std::fixed << std::setprecision(9)
                  << elapsed << "," << elapsed << ",0.000000000\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
