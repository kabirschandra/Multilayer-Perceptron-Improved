#include "mlp.h++"
#include <iostream>
#include <cmath>
#include <cstdio>
int main() {
    const char *path="mlp_save_load_test.bin";
    MLP::MLP a({2,5,2},{MLP::TANH,MLP::SIGMOID},{MLP::XAVIER_UNIFORM,MLP::XAVIER_UNIFORM});
    Eigen::MatrixXf x(2,3); x << 0,1,2, 3,4,5;
    Eigen::MatrixXf y=Eigen::MatrixXf::Zero(2,3);
    a.forward(x); a.backward(y,0.01f); a.forward(x);
    const float expected=a.mse(y);
    a.save(path);
    MLP::MLP b({2,2},{MLP::SIGMOID},{MLP::XAVIER_UNIFORM});
    b.load(path); b.forward(x);
    const float actual=b.mse(y);
    b.backward(y,0.01f); // cache must be restored after load
    std::remove(path);
    std::cout << "Save/load expected="<<expected<<" actual="<<actual<<"\n";
    return std::abs(expected-actual)<1e-6f ? 0:1;
}
