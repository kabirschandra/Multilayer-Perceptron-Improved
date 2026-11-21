#ifndef MLP_HPP
#define MLP_HPP
#include <Eigen/Dense>
#include "eigen_serialization.h++"
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>

namespace MLP {

    enum ACTIVATION_FUNCTION {
        RELU,
        SIGMOID,
        TANH,
    };

    enum INITIALISATION {
        UNIFORM,
        XAVIER_UNIFORM,
        HE_KAIMING,
    };


    namespace internal {

        class LayerParams {

            public:
            Eigen::MatrixXf w;
            Eigen::VectorXf b;
            Eigen::MatrixXf a;
            ACTIVATION_FUNCTION act;

            template<class Archive>
            void serialize(Archive &archive) {
                archive(w, b, act);
            }
        };
        class LayerCache {

            public:
            Eigen::MatrixXf dw;
            Eigen::VectorXf db;
            Eigen::MatrixXf dA;
            Eigen::MatrixXf dZ;
        };
    }


    class MLP {

        private:
        size_t batchSize;
        Eigen::MatrixXf x; //Network input
        std::vector<internal::LayerParams> layerParams;
        std::vector<internal::LayerCache> layerCache;



        public:
        MLP(const std::vector<size_t> &neurons, const std::vector<ACTIVATION_FUNCTION> &acts, const std::vector<INITIALISATION> &initialisations, const size_t batchSize = 1) {

            if((neurons.size() - 1 != acts.size()) || (acts.size() != initialisations.size())) {
                throw std::runtime_error("Dimensions of neurone and activation vectors are invalid. size(neurones) - 1 must equal size(acts), and size(acts) must equal size(initialisations). [neurones, acts, initialisations] = [" 
                    + std::to_string(neurons.size()) + ", " + std::to_string(acts.size()) + "," + std::to_string(initialisations.size()) + "]");
            }

            size_t nLayers = neurons.size() - 1;
            this->layerParams.resize(nLayers);
            this->layerCache.resize(nLayers);
            this->batchSize = batchSize; //Can change in forward/backward
            this->x = Eigen::MatrixXf(neurons.front(), batchSize);

            for(size_t i = 1; i <= nLayers; i++) {

                internal::LayerParams &currentParam = this->layerParams[i - 1];
                internal::LayerCache &currentCache = this->layerCache[i - 1];

                size_t dPrev = neurons[i - 1];
                size_t dCurrent = neurons[i];

                currentParam.w.resize(dCurrent, dPrev);
                currentParam.b.resize(dCurrent);
                currentParam.act = acts[i - 1];
                currentParam.a.resize(dCurrent, batchSize);

                currentCache.dw.resize(dCurrent, dPrev);
                currentCache.db.resize(dCurrent);
                currentCache.dA.resize(dCurrent, batchSize);
                currentCache.dZ.resize(dCurrent, batchSize);


                currentParam.b.setZero(); //Zero initialisation for biases
                switch(initialisations[i - 1]) {
                    case UNIFORM: {
                        currentParam.w.setRandom(); //Over -1, 1

                        break;
                    } case XAVIER_UNIFORM: {
                        float limit = std::sqrt(6.0f / (currentParam.w.rows() + currentParam.w.cols()));
                        currentParam.w.setRandom();
                        currentParam.w *= limit;

                        break;
                    } case HE_KAIMING: {
                        float limit = std::sqrt(6.0f / currentParam.w.cols());
                        currentParam.w.setRandom();
                        currentParam.w *= limit;

                        break;
                    }
                }
            }

        }


        void disp(void) {

            for(size_t i = 0; i < this->layerParams.size(); i++) {
                internal::LayerParams &currentParam = this->layerParams[i];
                std::cout << "Layer: " << i << "\n";

                std::cout << "Weight: \n";
                std::cout << currentParam.w << "\n";

                std::cout << "Bias: \n";
                std::cout << currentParam.b << "\n";

                std::cout << "Act: \n";
                std::cout << currentParam.act << "\n\n";
            }

            return;
        }

        void forward(const Eigen::MatrixXf &x) {

            if(x.rows() != this->layerParams.front().w.cols()) {
                throw std::runtime_error("Input matrix dimensions are incompatible with network. Network front contains " 
                    + std::to_string(this->layerParams.front().w.cols()) 
                    + " neurones, but input to network expects "
                    + std::to_string(x.rows()) + " neurones."
                );
            }

            this->x = x;
            for(size_t i = 0; i < this->layerParams.size(); i++) {

                Eigen::MatrixXf *xLayer;
                Eigen::MatrixXf &yLayer = this->layerParams[i].a;
                if(i == 0) {
                    xLayer = &(this->x);

                } else {
                    xLayer = &(this->layerParams[i - 1].a);
                }

                yLayer = (this->layerParams[i].w)*(*xLayer);
                yLayer.colwise() += this->layerParams[i].b;

                switch(this->layerParams[i].act) {
                    case RELU: {
                        yLayer = yLayer.cwiseMax(0.0f);

                        break;
                    } case TANH: {
                        yLayer = yLayer.array().tanh().matrix();

                        break;
                    } case SIGMOID: {
                        yLayer = (1.0f / (1.0f + (-yLayer.array()).exp())).matrix();

                        break;
                    }
                }


            }


            return;
        }


        void backward(const Eigen::MatrixXf yActual, const float lr) {

            if(yActual.cols() != this->layerParams.back().a.cols() ||
            yActual.rows() != this->layerParams.back().a.rows()) {
                throw std::runtime_error(
                    "Prediction and expected output dimensions do not match. "
                    "Expected [" + std::to_string(this->layerParams.back().a.rows()) + ", "
                    + std::to_string(this->layerParams.back().a.cols()) + "], got ["
                    + std::to_string(yActual.rows()) + ", "
                    + std::to_string(yActual.cols()) + "]."
                );
            }

            if(!std::isfinite(lr) || lr <= 0.0f) {
                throw std::runtime_error("Learning rate must be finite and greater than zero.");
            }





            

            return;
        }

        void save(const std::string &fileName) {
          
            std::ofstream file(fileName, std::ios::binary); 

            if(!file.is_open()) {
                throw std::runtime_error("Failed to open file for writing: " + fileName);
            }

            cereal::BinaryOutputArchive archive(file);
            
            archive(this->batchSize, this->x);
            archive(this->layerParams);
            
            return;
        }

        void load(const std::string &fileName) {

            std::ifstream file(fileName, std::ios::binary); 

            if(!file.is_open()) {
                throw std::runtime_error("Failed to open file for reading: " + fileName);
            }

            cereal::BinaryInputArchive archive(file);

            archive(this->batchSize, this->x);
            archive(this->layerParams);
            this->purge_cache(); //Purge existing cache

            return;
        }


        void purge_cache(void) {

            this->layerCache.clear();
            this->layerCache.shrink_to_fit();

            return;
        }

        void load_cache(void) {

            size_t batchSize = this->batchSize;
            this->layerCache.resize(this->layerParams.size());
            for(size_t i = 0; i < this->layerParams.size(); i++) {
                internal::LayerParams &currentParam = this->layerParams[i];

                //Caches
                internal::LayerCache &currentCache = this->layerCache[i];
                size_t dPrev = currentParam.w.cols();
                size_t dCurrent = currentParam.w.rows();


                currentCache.dw.resize(dCurrent, dPrev);
                currentCache.db.resize(dCurrent);
                currentCache.dA.resize(dCurrent, batchSize);
                currentCache.dZ.resize(dCurrent, batchSize);
            }

            return;
        }


    };
   
}



#endif





