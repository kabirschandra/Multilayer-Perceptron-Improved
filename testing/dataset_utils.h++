#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
namespace datasets {
namespace fs = std::filesystem;
inline fs::path root(){return fs::path(MLP_TEST_DATA_DIR);}
inline uint32_t read_u32(std::istream& in){unsigned char b[4];in.read(reinterpret_cast<char*>(b),4);if(!in)throw std::runtime_error("Truncated IDX header");return (uint32_t(b[0])<<24)|(uint32_t(b[1])<<16)|(uint32_t(b[2])<<8)|b[3];}
struct Digits {Eigen::MatrixXf x;std::vector<int> labels;};
inline Digits mnist(const std::string& images,const std::string& labels){
 std::ifstream im(root()/"MNIST"/images,std::ios::binary),la(root()/"MNIST"/labels,std::ios::binary);
 if(!im||!la)throw std::runtime_error("Missing MNIST files: run python3 testing/download_data.py");
 if(read_u32(im)!=2051||read_u32(la)!=2049)throw std::runtime_error("Invalid MNIST magic number");
 const auto n=read_u32(im),rows=read_u32(im),cols=read_u32(im),nl=read_u32(la);
 if(n!=nl||rows!=28||cols!=28||n==0||n>60000)throw std::runtime_error("Invalid MNIST dimensions");
 Digits d;d.x.resize(784,n);d.labels.resize(n);
 std::array<unsigned char,784> pixels{};
 for(uint32_t i=0;i<n;++i){im.read(reinterpret_cast<char*>(pixels.data()),784);if(!im)throw std::runtime_error("Truncated MNIST images");for(int p=0;p<784;++p)d.x(p,i)=pixels[p]/255.f;}
 for(uint32_t i=0;i<n;++i){char c;la.get(c);if(!la)throw std::runtime_error("Truncated MNIST labels");d.labels[i]=static_cast<unsigned char>(c);if(d.labels[i]>9)throw std::runtime_error("Invalid MNIST label");}
 if(im.peek()!=EOF||la.peek()!=EOF)throw std::runtime_error("Unexpected trailing MNIST data");
 return d;
}
struct Iris {Eigen::MatrixXf x;std::vector<int> labels;};
inline Iris iris(){
 std::ifstream in(root()/"IRIS"/"iris.data");if(!in)throw std::runtime_error("Missing Iris: run python3 testing/download_data.py");
 std::vector<std::array<float,4>> features;std::vector<int> labels;std::string line;
 while(std::getline(in,line)){
  if(line.empty()||line=="\r")continue;
  std::replace(line.begin(),line.end(),',',' ');std::istringstream ss(line);
  std::array<float,4> v{};std::string species;
  if(!(ss>>v[0]>>v[1]>>v[2]>>v[3]>>species))throw std::runtime_error("Malformed Iris row");
  int cls=species=="Iris-setosa"?0:species=="Iris-versicolor"?1:species=="Iris-virginica"?2:-1;
  if(cls<0)throw std::runtime_error("Unknown Iris class: "+species);
  features.push_back(v);labels.push_back(cls);
 }
 if(features.size()!=150)throw std::runtime_error("Expected 150 Iris rows");
 Iris d;d.x.resize(4,150);d.labels=std::move(labels);
 for(int i=0;i<150;++i)for(int j=0;j<4;++j)d.x(j,i)=features[i][j];
 return d;
}
inline Eigen::MatrixXf onehot(const std::vector<int>& labels,const std::vector<int>& indices,int classes){
 Eigen::MatrixXf y=Eigen::MatrixXf::Zero(classes,static_cast<Eigen::Index>(indices.size()));
 for(size_t j=0;j<indices.size();++j)y(labels[indices[j]],static_cast<Eigen::Index>(j))=1.f;
 return y;
}
inline Eigen::MatrixXf columns(const Eigen::MatrixXf& x,const std::vector<int>& ids){
 Eigen::MatrixXf result(x.rows(),static_cast<Eigen::Index>(ids.size()));
for(size_t j = 0; j < ids.size(); ++j) {
    result.col(static_cast<Eigen::Index>(j)) = x.col(ids[j]);
}
return result;
}
inline float accuracy(const Eigen::MatrixXf& predictions,const std::vector<int>& labels,const std::vector<int>& ids){
 int correct=0;for(size_t j=0;j<ids.size();++j){Eigen::Index idx;predictions.col(static_cast<Eigen::Index>(j)).maxCoeff(&idx);correct+=(idx==labels[ids[j]]);}return float(correct)/ids.size();
}
inline std::vector<int> range(int n){std::vector<int> ids(n);std::iota(ids.begin(),ids.end(),0);return ids;}
}
