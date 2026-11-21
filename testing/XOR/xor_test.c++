#include "mlp.h++"
#include <iostream>
#include <cmath>
int main() {
    MLP::MLP net({2,8,1},{MLP::TANH,MLP::SIGMOID},{MLP::XAVIER_UNIFORM,MLP::XAVIER_UNIFORM});
    Eigen::MatrixXf x(2,4), y(1,4);
    x << 0,0,1,1, 0,1,0,1;
    y << 0,1,1,0;
    net.forward(x);
    float before=net.mse(y);
    for(int epoch=0;epoch<12000;++epoch) {net.forward(x);net.backward(y,0.5f);}
    net.forward(x);
    float after=net.mse(y);
    std::cout << "XOR initial="<<before<<" final="<<after<<"\n";
    return (std::isfinite(after) && after<0.05f) ? 0:1;
}
