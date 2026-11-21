#ifndef MATRIX_IO_HPP
#define MATRIX_IO_HPP
#include <Eigen/Dense>
#include <cereal/cereal.hpp>
#include "eigen_serialization.h++"

#include <type_traits>
#include <string>
#include <fstream>

/**
 * @brief Saves an Eigen matrix to a binary file.
 * @tparam Matrix Eigen matrix type.
 * @param matrix Matrix to save.
 * @param filename Destination file path.
 */
template<typename Matrix>
    void save_matrix(const Matrix &matrix, const std::string &fileName) {
        std::ofstream file(fileName, std::ios::binary);

        if(!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + fileName);
        }

        cereal::BinaryOutputArchive archive(file);
        archive(matrix);

        return;
    }


/**
 * @brief Loads an Eigen matrix from a binary file.
 * @tparam Matrix Eigen matrix type.
 * @param matrix Matrix to populate.
 * @param filename Source file path.
 */
template<typename Matrix>
    void load_matrix(Matrix &matrix, const std::string &fileName) {

        std::ifstream file(file, std::ios::binary);

        if(!file.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + fileName);
        }

        cereal::BinaryInputArchive archive(file);
        archive(matrix);

        return;
    }



#endif
