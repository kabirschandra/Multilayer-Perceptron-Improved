#ifndef EIGEN_SERIALIZATION
#define EIGEN_SERIALIZATION
#include <Eigen/Dense>
#include <cereal/cereal.hpp>
#include <stdexcept>
#include <type_traits>


namespace cereal {





/**
 * @brief Serialises an Eigen matrix into a cereal binary archive.
 * @tparam Archive Cereal archive type.
 * @tparam Scalar Matrix element type.
 * @tparam nRows Compile-time number of rows.
 * @tparam nCols Compile-time number of columns.
 * @tparam options Eigen matrix storage options.
 * @tparam maxRows Maximum compile-time number of rows.
 * @tparam maxCols Maximum compile-time number of columns.
 * @param archive Output archive.
 * @param matrix Matrix to serialise.
 */
template<class Archive, typename Scalar, int nRows, int nCols, int options, int maxRows, int maxCols>
    void save(Archive &archive, const Eigen::Matrix<Scalar, nRows, nCols, options, maxRows, maxCols> &matrix) {
        std::int64_t rows = matrix.rows();
        std::int64_t cols = matrix.cols();

        archive(rows, cols);
        archive(cereal::binary_data(matrix.data(), static_cast<std::size_t>(matrix.size()) * sizeof(Scalar)));

        return;
    }



/**
 * @brief Deserialises an Eigen matrix from a cereal binary archive.
 * @tparam Archive Cereal archive type.
 * @tparam Scalar Matrix element type.
 * @tparam nRows Compile-time number of rows.
 * @tparam nCols Compile-time number of columns.
 * @tparam options Eigen matrix storage options.
 * @tparam maxRows Maximum compile-time number of rows.
 * @tparam maxCols Maximum compile-time number of columns.
 * @param archive Input archive.
 * @param matrix Matrix to populate with deserialised data.
 * @throws std::runtime_error If the matrix dimensions are invalid or incompatible.
 */
template<class Archive, typename Scalar, int nRows, int nCols, int options, int maxRows, int maxCols>
    void load(Archive &archive, Eigen::Matrix<Scalar, nRows, nCols, options, maxRows, maxCols> &matrix) {
        std::int64_t rows;
        std::int64_t cols;

        archive(rows, cols);

        constexpr std::int64_t maxElements = 1e8;

        if (rows < 0 || cols < 0 ||
            (rows != 0 && cols > maxElements / rows) ||
            (nRows != Eigen::Dynamic && rows != nRows) ||
            (nCols != Eigen::Dynamic && cols != nCols) ||
            (maxRows != Eigen::Dynamic && rows > maxRows) ||
            (maxCols != Eigen::Dynamic && cols > maxCols)) {
            throw std::runtime_error("Invalid Eigen matrix dimensions");
        }

        matrix.resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));

        archive(cereal::binary_data(matrix.data(), static_cast<std::size_t>(matrix.size()) * sizeof(Scalar)));

        return;
    }

}






#endif
