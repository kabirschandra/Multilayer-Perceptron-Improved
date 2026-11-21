# C++ Multilayer Perceptron Improved 

A C++17 CPU-based, multilayer feedforward neural network using Eigen matrices and cereal binary serialization. Supports configurable dense layers, ReLU/tanh/sigmoid activations, uniform/Xavier/He initialisation, minibatch backpropagation, MSE loss, and saving/loading weights.

## Dependencies

- C++17 compiler (GCC or Clang), CMake 3.16+, Python 3 (standard library only for downloading test datasets).
- `external/eigen/Eigen/Dense` and `external/cereal-1.3.2/include/cereal/cereal.hpp` must be present. Keep your existing vendored dependencies; they are **not included** in the package.
- Linux/WSL recommended.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMLP_NATIVE=ON -DMLP_FAST_MATH=OFF
cmake --build build -j$(nproc)
./build/main
ctest --test-dir build --output-on-failure
```

`ctest` automatically downloads missing Iris/MNIST data and validates it. First run requires internet; subsequent runs use cached files. Alternatively:

```bash
cmake --build build --target download_datasets
./testing/run_all.sh
```

Run one test: `ctest --test-dir build -R '^iris$' --output-on-failure` or `./build/test_iris` (after downloading data). Run MNIST with optional epoch count and training sample limit: `./build/test_mnist 5 60000`. The default is 3 epochs over 60,000 training examples. Actual training time depends on CPU.

## Optimisation

Release uses `-O3`, `-DNDEBUG`, supported LTO, and optional `-march=native` (`MLP_NATIVE=ON`). `MLP_FAST_MATH=ON` enables `-ffast-math`, which can change numerical behaviour. Prefer OFF for correctness tests. Debug/ASan builds should disable fast math.

## Tests

| Test | Source | Behaviour |
| --- | --- | --- |
| XOR | `testing/XOR/xor_test.c++` | Learns the four XOR samples |
| Save/load | `testing/Save_load/save_load_test.c++` | Round-trip cereal serialization, verifies output and continued training |
| Iris | `testing/IRIS/test_iris.c++` | Reads real 150-row UCI Iris data, stratified 120/30 train/test split, reports held-out accuracy and MSE |
| MNIST | `testing/MNIST/test_mnist.c++` | Reads actual IDX training/test files, trains on real images, evaluates all 10,000 held-out images |

The dataset test checks source availability and file structure. Iris accuracy threshold is 80%; MNIST threshold is 65%. These are basic functional thresholds, **not** state-of-the-art accuracy targets. The data are not synthetic. Weight initialisation is random, so results may vary across runs.

## Data layout

```text
testing/
  download_data.py
  dataset_utils.h++
  run_all.sh
  XOR/xor_test.c++
  Save_load/save_load_test.c++
  IRIS/iris.data
  IRIS/test_iris.c++
  MNIST/train-images-idx3-ubyte
  MNIST/train-labels-idx1-ubyte
  MNIST/t10k-images-idx3-ubyte
  MNIST/t10k-labels-idx1-ubyte
  MNIST/test_mnist.c++
```

Dataset downloads are cached locally; missing or invalid files are downloaded again. The download script validates MNIST gzip SHA-256 checksums and IDX dimensions. Iris is the UCI 150-row original dataset.

Sources: [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) and [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist).

## Example

```cpp
#include "mlp.h++"
int main() {
    MLP::MLP net({2,8,1}, {MLP::TANH,MLP::SIGMOID},
                 {MLP::XAVIER_UNIFORM,MLP::XAVIER_UNIFORM});
    Eigen::MatrixXf x(2,4), y(1,4);
    x << 0,0,1,1, 0,1,0,1;
    y << 0,1,1,0;
    for(int i=0;i<1000;++i) {net.forward(x);net.backward(y,0.1f);}
    net.forward(x);
    float loss=net.mse(y);
    const Eigen::MatrixXf& predictions=net.output();
    (void)loss;(void)predictions;
}
```

`forward()` accepts matrices `[input features, batch size]`; columns are individual examples. `output()` returns the most recent activated output matrix. `save()` and `load()` use cereal binary archives; run `forward()` after loading to recreate activations before training. Gradient cache is restored after loading.

## Licenses and Acknowledgements

This project uses third-party open-source libraries and publicly available machine-learning datasets. The respective authors and copyright holders retain ownership of their work.

### Software dependencies

**Eigen — Mozilla Public License 2.0 (MPL-2.0)**

Eigen is a C++ template library for linear algebra, developed by the Eigen project and its contributors. This project uses Eigen for matrix operations, vector operations, and neural network computations.

- Project: https://eigen.tuxfamily.org/
- Source: https://gitlab.com/libeigen/eigen
- License: https://www.mozilla.org/en-US/MPL/2.0/
- License identifier: `MPL-2.0`

Eigen versions 3.1.1 onward are primarily licensed under MPL-2.0. Some optional components may have additional licensing requirements. Users redistributing Eigen should retain its original copyright and license notices and comply with the applicable license terms.

**cereal — BSD 3-Clause License**

cereal is a header-only C++ serialization library developed by Shane Grant, Randolph Voorhies, and contributors to the USCiLab project. This project uses cereal 1.3.2 for binary serialization and deserialization of neural network parameters.

- Project: https://uscilab.github.io/cereal/
- Source: https://github.com/USCilab/cereal
- License: https://opensource.org/licenses/BSD-3-Clause
- License identifier: `BSD-3-Clause`

The BSD 3-Clause License permits redistribution and modification, subject to retaining the required copyright notices, license conditions, and disclaimers.

### Machine-learning datasets

**MNIST — Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)**

The MNIST database of handwritten digits was created by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges, using handwritten-digit data derived from the United States National Institute of Standards and Technology (NIST).

This project uses MNIST exclusively for training, testing, and benchmarking its multilayer perceptron implementation.

- Original dataset: https://yann.lecun.com/exdb/mnist/
- Dataset mirror: https://github.com/cvdfoundation/mnist
- License: https://creativecommons.org/licenses/by-sa/3.0/
- License identifier: `CC-BY-SA-3.0`

**Citation:**

LeCun, Y., Cortes, C., & Burges, C. J. C. *The MNIST Database of Handwritten Digits.*

Related publication:

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278–2324. https://doi.org/10.1109/5.726791

MNIST remains the property of its respective copyright holders. Redistribution and adaptation must comply with the applicable Creative Commons attribution and ShareAlike requirements.

**Iris — Creative Commons Attribution 4.0 (CC BY 4.0)**

The Iris dataset originates from the work of Ronald A. Fisher and is distributed through the UCI Machine Learning Repository.

The dataset contains 150 flower observations, each with four numerical measurements and a species classification. It is used in this project to evaluate multiclass classification performance.

- Dataset: https://archive.ics.uci.edu/dataset/53/iris
- DOI: https://doi.org/10.24432/C56C76
- License: https://creativecommons.org/licenses/by/4.0/
- License identifier: `CC-BY-4.0`

**Citation:**

Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76

Original publication:

Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179–188.

The Iris dataset is distributed under CC BY 4.0, which permits reuse and adaptation with appropriate attribution.

### Redistribution and third-party notices

- The Eigen and cereal libraries are third-party dependencies and are not original components of this project's neural network implementation.
- Any redistributed copies of these libraries must retain the applicable license texts, copyright notices, and required disclaimers.
- The MNIST and Iris datasets are downloaded separately for testing. They are not original works of this project.
- Dataset attribution and licensing obligations remain applicable if downloaded datasets are redistributed.
- The software licenses and dataset licenses apply to their respective components; they do not automatically determine the license of the original MLP implementation.

Thanks to the Eigen and cereal contributors, the MNIST creators, R. A. Fisher, the UCI Machine Learning Repository, NIST, and the CVDF MNIST mirror maintainers for making these resources available.
