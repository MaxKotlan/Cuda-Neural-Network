#include <iostream>
#include <vector>
#include <Cuda-Convolution-Benchmark/Convolution.h>
#include "idx.h"

int main(int argc, char** argv){
    Result<int> r = CudaPerformConvolution<int,0>(std::vector<int>{0}, std::vector<int>{0}, getKernels<int,0>()[0]);
    IDX::ImageDatabase t10k("data/t10k-images.idx3-ubyte");
    std::vector<unsigned char> image_data = t10k.GetImage(567);
    for (auto it = image_data.begin(); it != image_data.end(); it++)
        std::cout << (int)*it;
}