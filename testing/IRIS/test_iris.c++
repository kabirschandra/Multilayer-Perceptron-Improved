#include "mlp.h++"
#include "dataset_utils.h++"
#include <iostream>
#include <random>
int main(){try{
 auto data=datasets::iris();auto ids=datasets::range(150);std::mt19937 rng(42);
 // Stratified 80/20 split: 40 training and 10 test samples per species.
 std::vector<int> train,test;
 for(int cls=0;cls<3;++cls){std::vector<int> group;for(int i:ids)if(data.labels[i]==cls)group.push_back(i);std::shuffle(group.begin(),group.end(),rng);train.insert(train.end(),group.begin(),group.begin()+40);test.insert(test.end(),group.begin()+40,group.end());}
 std::shuffle(train.begin(),train.end(),rng);
 // Normalise using training statistics only to prevent test leakage.
 Eigen::VectorXf mean=Eigen::VectorXf::Zero(4),sd=Eigen::VectorXf::Zero(4);
for(int i : train) {
    mean += data.x.col(i);
}
mean /= static_cast<float>(train.size());
for(int i : train) {
    sd += (data.x.col(i) - mean).array().square().matrix();
}

sd = (sd / static_cast<float>(train.size())).array().sqrt();
 for(int r=0;r<4;++r)if(sd(r)<1e-6f)sd(r)=1.f;
 for(int c=0;c<data.x.cols();++c)data.x.col(c)=(data.x.col(c)-mean).cwiseQuotient(sd);
 MLP::MLP net({4,16,3},{MLP::TANH,MLP::SIGMOID},{MLP::XAVIER_UNIFORM,MLP::XAVIER_UNIFORM});
 constexpr int batch=20;
 for(int epoch=0;epoch<1200;++epoch){std::shuffle(train.begin(),train.end(),rng);for(size_t k=0;k<train.size();k+=batch){std::vector<int> subset(train.begin()+k,train.begin()+std::min(train.size(),k+batch));auto x=datasets::columns(data.x,subset);auto y=datasets::onehot(data.labels,subset,3);net.forward(x);net.backward(y,0.12f);}}
 net.forward(datasets::columns(data.x,test));
 float acc=datasets::accuracy(net.output(),data.labels,test);
 auto y=datasets::onehot(data.labels,test,3);float loss=net.mse(y);
 std::cout<<"Iris held-out accuracy="<<acc*100.f<<"% MSE="<<loss<<" (30 samples)\n";
 return std::isfinite(loss)&&acc>=0.80f?0:1;
 }catch(const std::exception&e){std::cerr<<"Iris FAIL: "<<e.what()<<"\n";return 1;}}
