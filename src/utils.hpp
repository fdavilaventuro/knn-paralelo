#ifndef KNN_UTILS_HPP
#define KNN_UTILS_HPP

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace knn {

inline void read_csv(const std::string& filename,
                     std::vector<double>& features,
                     std::vector<int>& targets,
                     std::size_t& feature_count) {
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error("No se pudo abrir el archivo CSV: " + filename);
    }

    features.clear();
    targets.clear();
    feature_count = 0;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line == "\r") {
            continue;
        }

        std::stringstream row_stream(line);
        std::string field;
        std::vector<double> values;

        while (std::getline(row_stream, field, ',')) {
            if (field.empty()) {
                throw std::runtime_error("Campo vacío en " + filename +
                                         " (línea " + std::to_string(line_number) + ")");
            }

            try {
                std::size_t parsed_chars = 0;
                const double value = std::stod(field, &parsed_chars);
                if (parsed_chars != field.size() && field[parsed_chars] != '\r') {
                    throw std::invalid_argument("caracteres adicionales");
                }
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("valor no finito");
                }
                values.push_back(value);
            } catch (const std::exception&) {
                throw std::runtime_error("Valor inválido en " + filename +
                                         " (línea " + std::to_string(line_number) + ")");
            }
        }

        if (values.size() < 2) {
            throw std::runtime_error("Cada fila debe tener al menos un feature y un target en " +
                                     filename + " (línea " + std::to_string(line_number) + ")");
        }

        const std::size_t current_feature_count = values.size() - 1;
        if (feature_count == 0) {
            feature_count = current_feature_count;
        } else if (feature_count != current_feature_count) {
            throw std::runtime_error("Número inconsistente de features en " + filename +
                                     " (línea " + std::to_string(line_number) + ")");
        }

        const double target_value = values.back();
        if (std::floor(target_value) != target_value ||
            target_value < std::numeric_limits<int>::min() ||
            target_value > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Target inválido en " + filename +
                                     " (línea " + std::to_string(line_number) + ")");
        }

        features.insert(features.end(), values.begin(), values.end() - 1);
        targets.push_back(static_cast<int>(target_value));
    }

    if (targets.empty()) {
        throw std::runtime_error("El archivo CSV no contiene datos: " + filename);
    }
}

}  // namespace knn

#endif
