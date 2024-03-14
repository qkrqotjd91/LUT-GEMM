# LUT-GEMM

## Quick Start

Run the following commands to get "Kernel Evaluation" results in Table 1.

``` sh
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
make -j8
./tests/tests  
```