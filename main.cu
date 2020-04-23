#include <iostream>
#include <vector>
#include <Cuda-Convolution-Benchmark/Convolution.h>
#include "idx.h"

int main(int argc, char** argv){
    Result<int> r = CudaPerformConvolution<int,0>(std::vector<int>{0}, std::vector<int>{0}, getKernels<int,0>()[0]);
    IDX::ImageDatabase("data/t10k-images.idx3-ubyte");
    std::cout << "Hello World!" << std::endl;
}