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
            Eigen::MatrixXf da;
            Eigen::MatrixXf dz;
        };
    }


    class MLP {

        private:
        size_t batchSize;
        Eigen::MatrixXf x; //Network input
        std::vector<internal::LayerParams> layerParams;
        std::vector<internal::LayerCache> layerCache;



        public:
        /**
         * @brief Constructs a multilayer perceptron with the specified architecture.
         * @param neurons Number of neurons in each layer, including the input layer.
         * @param acts Activation function for each non-input layer.
         * @param initialisations Weight initialisation method for each non-input layer.
         * @throws std::runtime_error If the architecture dimensions are inconsistent.
         */
        MLP(const std::vector<size_t> &neurons, const std::vector<ACTIVATION_FUNCTION> &acts, const std::vector<INITIALISATION> &initialisations) {

            if((neurons.size() - 1 != acts.size()) || (acts.size() != initialisations.size())) {
                throw std::runtime_error("Dimensions of neurone and activation vectors are invalid. size(neurones) - 1 must equal size(acts), and size(acts) must equal size(initialisations). [neurones, acts, initialisations] = [" 
                    + std::to_string(neurons.size()) + ", " + std::to_string(acts.size()) + "," + std::to_string(initialisations.size()) + "]");
            }

            size_t nLayers = neurons.size() - 1;
            this->batchSize = 1;
            this->layerParams.resize(nLayers);
            this->layerCache.resize(nLayers);
            this->x = Eigen::MatrixXf(neurons.front(), this->batchSize);
            for(size_t i = 1; i <= nLayers; i++) {

                internal::LayerParams &currentParam = this->layerParams[i - 1];
                internal::LayerCache &currentCache = this->layerCache[i - 1];

                size_t dPrev = neurons[i - 1];
                size_t dCurrent = neurons[i];

                currentParam.w.resize(dCurrent, dPrev);
                currentParam.b.resize(dCurrent);
                currentParam.act = acts[i - 1];
                currentParam.a.resize(dCurrent, this->batchSize);

                currentCache.dw.resize(dCurrent, dPrev);
                currentCache.db.resize(dCurrent);
                currentCache.da.resize(dCurrent, this->batchSize);
                currentCache.dz.resize(dCurrent, this->batchSize);


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

        /**
         * @brief Displays the network's weights, biases and activation functions.
         */
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


        /**
         * @brief Performs forward propagation through the network.
         * @param x Input matrix with dimensions [input neurons, batch size].
         *          Each column represents one sample.
         * @throws std::runtime_error If the input dimensions are invalid or the batch is empty.
         * @note Updates the stored batch size and layer activations.
         */
        void forward(const Eigen::MatrixXf &x) {

            if(x.rows() != this->layerParams.front().w.cols()) {
                throw std::runtime_error("Input matrix dimensions are incompatible with network. Network front contains " 
                    + std::to_string(this->layerParams.front().w.cols()) 
                    + " neurones, but input to network expects "
                    + std::to_string(x.rows()) + " neurones."
                );
            }
            if(x.cols() == 0) {
                throw std::runtime_error("Batch must be greater than zero.");
            }  

            this->batchSize = x.cols();
            this->x = x;
            for(size_t i = 0; i < this->layerParams.size(); i++) {

                Eigen::MatrixXf *xLayer;
                Eigen::MatrixXf &yLayer = this->layerParams[i].a;
                if(i == 0) {
                    xLayer = &(this->x);

                } else {
                    xLayer = &(this->layerParams[i - 1].a);
                }

                yLayer.noalias() = this->layerParams[i].w * (*xLayer);
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



        /**
         * @brief Performs backpropagation and updates the network parameters.
         * @param yActual Expected outputs with dimensions [output neurons, batch size].
         * @param lr Learning rate for gradient descent.
         * @throws std::runtime_error If output dimensions or batch size do not match,
         *         or if the learning rate is invalid.
         * @note Requires a preceding forward() call and an allocated gradient cache.
         * @note Assumes the output gradient is derived from half-squared-error loss.
         */
        void backward(const Eigen::MatrixXf &yActual, const float lr) {

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

            if(yActual.cols() != static_cast<Eigen::Index>(this->batchSize)) {
                throw std::runtime_error("Batch size mismatch. [batch size, yActual.cols()]: [" + std::to_string(this->batchSize) + ", " + std::to_string(yActual.cols()) + "]\n");
            }

            if(!std::isfinite(lr) || lr <= 0.0f) {
                throw std::runtime_error("Learning rate must be finite and greater than zero.");
            }


            for(size_t i = this->layerParams.size(); i-- > 0; ) {
                internal::LayerParams &currentParam = this->layerParams[i];
                internal::LayerCache &currentCache = this->layerCache[i];


               
                if(i == this->layerParams.size() - 1) { //Final layer
                    currentCache.da = currentParam.a - yActual;          
                } else { //Non last layer
                    currentCache.da.noalias() = this->layerParams[i + 1].w.transpose() * this->layerCache[i + 1].dz;
                }


                switch(currentParam.act) {
                    case RELU: {
                        currentCache.dz = currentCache.da.array() * (currentParam.a.array() > 0.0f).cast<float>();
                        break;
                    } case TANH: {
                        currentCache.dz = currentCache.da.array() * (1.0f - currentParam.a.array().square());
                        break;
                    } case SIGMOID: {
                        currentCache.dz = currentCache.da.array() * (currentParam.a.array()) * (1.0f - currentParam.a.array());
                        break;
                    }
                }

                const Eigen::MatrixXf &aPrev =
                    (i == 0) ? this->x : this->layerParams[i - 1].a;

                currentCache.dw.noalias() =
                    currentCache.dz * aPrev.transpose();

                currentCache.dw /= static_cast<float>(this->batchSize);

                currentCache.db.noalias() = (1.0f/this->batchSize) * (currentCache.dz.rowwise().sum());

            }

            //Update weights
            for(size_t i = this->layerParams.size(); i-- > 0; ) {
                internal::LayerParams &currentParam = this->layerParams[i];
                internal::LayerCache &currentCache = this->layerCache[i];

                currentParam.w = currentParam.w - (lr * currentCache.dw);
                currentParam.b = currentParam.b - (lr * currentCache.db);
            }


            return;
        }



        /**
         * @brief Calculates the mean squared error between predictions and targets.
         * @param yActual Expected outputs with dimensions [output neurons, batch size].
         * @return Mean squared error averaged across all output neurons and samples.
         * @throws std::runtime_error If output dimensions or batch size do not match.
         */
        float mse(const Eigen::MatrixXf &yActual) {
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

            if(yActual.cols() != static_cast<Eigen::Index>(this->batchSize)) {
                throw std::runtime_error("Batch size mismatch. [batch size, yActual.cols()]: [" + std::to_string(this->batchSize) + ", " + std::to_string(yActual.cols()) + "]\n");
            }

            const Eigen::MatrixXf &a = this->layerParams.back().a;

            return (a - yActual).array().square().mean();
        }

        /**
         * @brief Returns the output activations from the most recent forward pass.
         * @return Const reference to the output matrix.
         *         Rows represent output neurons; columns represent samples.
         */
        const Eigen::MatrixXf &output() const {
            return this->layerParams.back().a;
        }

        /**
         * @brief Saves the network state to a binary file using cereal.
         * @param fileName Path to the destination file.
         * @throws std::runtime_error If the file cannot be opened.
         * @note Saves the batch size, input matrix, weights, biases and activations'
         *         function types. Does not save gradient caches or layer activations.
         */
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


        /**
         * @brief Loads a previously saved network from a binary file.
         * @param fileName Path to the saved network file.
         * @throws std::runtime_error If the file cannot be opened.
         * @note Clears the existing gradient cache after loading.
         *       Call load_cache() before performing backpropagation.
         */
        void load(const std::string &fileName) {

            std::ifstream file(fileName, std::ios::binary); 

            if(!file.is_open()) {
                throw std::runtime_error("Failed to open file for reading: " + fileName);
            }

            cereal::BinaryInputArchive archive(file);

            archive(this->batchSize, this->x);
            archive(this->layerParams);
            this->purge_cache(); //Purge existing cache
            this->load_cache();
            return;
        }



        /**
         * @brief Clears the gradient cache and requests release of its allocated memory.
         * @note Does not modify the network weights or biases.
         */
        void purge_cache(void) {

            this->layerCache.clear();
            this->layerCache.shrink_to_fit();

            return;
        }




        /**
         * @brief Allocates or resizes gradient caches to match the network architecture.
         * @note Uses the currently stored batch size.
         * @note Does not initialise gradient values.
         */
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
                currentCache.da.resize(dCurrent, batchSize);
                currentCache.dz.resize(dCurrent, batchSize);
            }

            return;
        }


    };
   
}



#endif





