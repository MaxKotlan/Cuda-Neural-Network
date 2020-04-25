#include <iostream>
#include <vector>
#include  <iomanip>
#include <Cuda-Convolution-Benchmark/Convolution.h>
#include "idx.h"

int main(int argc, char** argv){
    Result<int> r = CudaPerformConvolution<int,0>(std::vector<int>{0}, std::vector<int>{0}, getKernels<int,0>()[0]);
    IDX::ImageDatabase t10k("data/t10k-images.idx3-ubyte");
    IDX::LabelDatabase t10klab("data/t10k-labels.idx1-ubyte");
    IDX::ImageDatabase t10ktrain("data/train-images.idx3-ubyte");
    IDX::LabelDatabase t10ktrainlab("data/train-labels.idx1-ubyte");

    for (int k = 0; k < 1000; k++){
        Image image_data = t10k.GetImage(k);
        std::cout << std::endl << std::endl << "This image is a " << t10klab.GetLabel(k) << std::endl;
        auto normalized = image_data.Normalize();
        for (int i = 0; i < image_data.size(); i++){
            if (i%image_data.x()==0) std::cout << std::endl;
            std::cout << std::fixed << std::setprecision(2) << /* std::hex<< std::setfill('0') << std::setw(2) <<*/ normalized[i];
        }
    }
}