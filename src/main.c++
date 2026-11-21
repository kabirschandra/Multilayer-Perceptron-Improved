#include "mlp.h++"
#include <iostream>


int main(void) {


    std::vector<size_t> dims = {5,4,3,2};
    std::vector<MLP::ACTIVATION_FUNCTION> act = {
        MLP::ACTIVATION_FUNCTION::RELU,
        MLP::ACTIVATION_FUNCTION::TANH,
        MLP::ACTIVATION_FUNCTION::SIGMOID,
    
    };
    std::vector<MLP::INITIALISATION> init = {
        MLP::INITIALISATION::HE_KAIMING,
        MLP::INITIALISATION::UNIFORM,
        MLP::INITIALISATION::XAVIER_UNIFORM,
    };

    MLP::MLP mlp(dims, act, init);
    mlp.disp(); 

    return 0;
}






