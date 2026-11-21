#ifndef MLP_HPP
#define MLP_HPP
#include <Eigen/Dense>
#include "eigen_serialization.h++"
#include "matrix_io.h++"
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

namespace MLP {

    enum ACTIVATION_FUNCTION {
        RELU,
        SIGMOID,
        TANH,
    };



    namespace internal {

        class LayerParams {

            public:
            Eigen::MatrixXf w;
            Eigen::VectorXf b;
            ACTIVATION_FUNCTION act;

        };
        class LayerCache {

            public:
            Eigen::MatrixXf dw;
            Eigen::VectorXf db;
            Eigen::MatrixXf a;
            Eigen::MatrixXf dA;
            Eigen::MatrixXf dZ;
        };
    }


    class MLP {

        private:
        Eigen::MatrixXf x; //Network input
        Eigen::MatrixXf y; //Network output
        std::vector<internal::LayerParams> layerParams;
        std::vector<internal::LayerCache> layerCache;



        public:
        MLP() {
            //todo
        }


        void forward(Eigen::MatrixXf &x) {

            if(x.rows() != this->layerParams.front().w.cols()) {
                throw std::runtime_error("Input matrix dimensions are incompatible with network. Network front contains " 
                    + std::to_string(this->layerParams.front().w.cols()) 
                    + " neurones, but input to network expects "
                    + std::to_string(x.rows()) + " neurones."
                );
            }

            //todo

            return;
        }


        void backward() {

        }

        void save() {

        }

        void load() {

        }



    };
   
}



#endif





