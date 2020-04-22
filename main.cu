#include <iostream>
#include <vector>
#include <Cuda-Convolution-Benchmark/Convolution.h>

int main(int argc, char** argv){
    //TestAllKernels(std::vector<float>{0,0}, std::vector<float>{1,1,1}, std::vector<float>{0,0,0,0});
    Result<int> r = CudaPerformConvolution<int,0>(std::vector<int>{0}, std::vector<int>{0}, getKernels<int,0>()[0]);
    std::cout << "Hello World!" << std::endl;
}