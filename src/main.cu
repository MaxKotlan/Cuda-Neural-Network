#include <iostream>
#include <vector>
#include  <iomanip>
#include "idx-cuda.h"
#include "neuralnetwork.h"
#include <thrust/device_vector.h>

int main(int argc, char** argv){
    
    IDX::CudaImageDatabase t10k("../data/t10k-images.idx3-ubyte");
    IDX::LabelDatabase     t10klab("../data/t10k-labels.idx1-ubyte");
    IDX::CudaImageDatabase t10ktrain("../data/train-images.idx3-ubyte");
    IDX::LabelDatabase     t10ktrainlab("../data/train-labels.idx1-ubyte");
    //std::cout << "TESTTING TESTING";
    //IDX::CudaImageDatabase testDB("../data/train-images.idx3-ubyte");
    /*
    for (int k = 0; k < 2; k++){
        Image image_data = t10k.GetImage(k);
        std::cout << std::endl << std::endl << "This image is a " << t10klab.GetLabel(k) << std::endl;
        auto normalized = image_data.Normalize();
        for (int i = 0; i < image_data.size(); i++){
            if (i%image_data.x()==0) std::cout << std::endl;
            std::cout << /*std::fixed << std::setprecision(2) <<*/ /* std::hex<< std::setfill('0') << std::setw(2) <<*//* image_data[i];
        }
    }*/

    srand(132);
    NeuralNetwork mynn(28*28, 16, 2, 10, 1.0);
    std::cout << std::hex<< std::setfill('0') << std::setw(2);///<< std::fixed << std::setprecision(6);
    //for (auto layer : mynn._layers){
    //    for (int j = 0; j < layer.outputsize; j++){
    //        std::cout << "[ ";
    //        for (int i = 0; i < layer.inputsize; i++)
    //            std::cout << layer.weights[i+j*layer.inputsize];
    //        std::cout << " ][ " << "?" << " ] + [ " << layer.biases[j] << "]" << std::endl;
    //    }
    //}
    
    for (int i = 0; i < 1000; i++){
        thrust::device_vector<unsigned char> image = t10k.GetImage(i).Normalize();
        std::vector<unsigned char> image_norm(image.size());
        thrust::copy(image.begin(), image.end(), image_norm.begin());
        for (int ch = 0; ch < image_norm.size(); ch++){
            if (ch%28 == 0) std::cout << std::endl;
            std::cout << std::hex<< std::setfill('0') << std::setw(2) << (int)image_norm[ch];
        }
    }

    /*
    int count = 0;
    while (true){
        thrust::device_vector<float> image = t10k.GetImage(0).Normalize();
        uint32_t label = t10klab.GetLabel(0);    
    //for (int i = 0; i < 1000; i++){
        std::cout << "Image " << 0 << ": Output Neurons: ";
        thrust::device_vector<float> result = mynn(image);
        std::cout << "Correct: " << label << " ";
        for (auto e : sresult)
            std::cout << e << ", ";
        mynn.TrainSingle(image, label);
        std::cout << std::endl;
        count++;
    //}
    }*/
}