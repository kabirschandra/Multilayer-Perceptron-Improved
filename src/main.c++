#include "mlp.h++"
#include <iostream>


int main(void) {

    Eigen::MatrixXd A(2, 2);

    A << 1, 2, 3, 4;

    {
        std::ofstream file("matrix.bin", std::ios::binary);
        cereal::BinaryOutputArchive archive(file);
        archive(A);
    }


    Eigen::MatrixXd matrix2;

    {
        std::ifstream file2("matrix.bin", std::ios::binary);
        cereal::BinaryInputArchive archive(file2);
        archive(matrix2);
    }

    std::cout << A << std::endl;

    return 0;
}






