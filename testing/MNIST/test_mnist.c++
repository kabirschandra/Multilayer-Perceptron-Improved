#include "mlp.h++"
#include "dataset_utils.h++"
#include <iostream>
#include <random>
#include <cstdlib>
int main(int argc,char**argv){try{
 auto train=datasets::mnist("train-images-idx3-ubyte","train-labels-idx1-ubyte");
 auto test=datasets::mnist("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte");
 if(train.x.cols()!=60000||test.x.cols()!=10000)throw std::runtime_error("Unexpected MNIST counts");
 int epochs=argc>1?std::stoi(argv[1]):3;int limit=argc>2?std::stoi(argv[2]):60000;
 if(epochs<1||limit<1||limit>60000)throw std::runtime_error("Usage: test_mnist [epochs>=1] [training_samples 1..60000]");
 MLP::MLP net({784,128,64,10},{MLP::RELU,MLP::RELU,MLP::SIGMOID},{MLP::HE_KAIMING,MLP::HE_KAIMING,MLP::XAVIER_UNIFORM});
 std::mt19937 rng(42);auto ids=datasets::range(60000);constexpr int batch=128;
 for(int epoch=0;epoch<epochs;++epoch){std::shuffle(ids.begin(),ids.end(),rng);double loss=0;int batches=0;
  for(int k=0;k<limit;k+=batch){int end=std::min(limit,k+batch);std::vector<int> subset(ids.begin()+k,ids.begin()+end);
   auto x=datasets::columns(train.x,subset);auto y=datasets::onehot(train.labels,subset,10);
   net.forward(x);loss+=net.mse(y);++batches;net.backward(y,0.18f);
  }
  std::cout<<"MNIST epoch "<<epoch+1<<"/"<<epochs<<" train MSE="<<loss/batches<<"\n";
 }
 int correct=0;double testLoss=0;int testBatches=0;
 for(int k=0;k<10000;k+=256){std::vector<int> idsTest;for(int j=k;j<std::min(k+256,10000);++j)idsTest.push_back(j);
  net.forward(datasets::columns(test.x,idsTest));
  auto y=datasets::onehot(test.labels,idsTest,10);float loss=net.mse(y);
  if(!std::isfinite(loss))throw std::runtime_error("Non-finite MNIST test loss");
  testLoss+=loss;++testBatches;
  for(size_t j=0;j<idsTest.size();++j){Eigen::Index predicted;net.output().col(static_cast<Eigen::Index>(j)).maxCoeff(&predicted);correct+=(predicted==test.labels[idsTest[j]]);}
 }
 float acc=correct/10000.f;
 std::cout<<"MNIST held-out accuracy="<<acc*100.f<<"% ("<<correct<<"/10000) test MSE="<<testLoss/testBatches<<"\n";
 return acc>=0.65f?0:1;
 }catch(const std::exception&e){std::cerr<<"MNIST FAIL: "<<e.what()<<"\n";return 1;}}
