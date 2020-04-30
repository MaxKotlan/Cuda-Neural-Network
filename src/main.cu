#include <iostream>
#include <vector>
#include  <iomanip>
#include "idx.h"
#include "neuralnetwork.h"

int main(int argc, char** argv){
    
    IDX::ImageDatabase t10k("../data/t10k-images.idx3-ubyte");
    IDX::LabelDatabase t10klab("../data/t10k-labels.idx1-ubyte");
    IDX::ImageDatabase t10ktrain("../data/train-images.idx3-ubyte");
    IDX::LabelDatabase t10ktrainlab("../data/train-labels.idx1-ubyte");

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
    NeuralNetwork mynn(28*28, 16, 2, 10);
    std::cout << std::fixed << std::setprecision(2);
    //for (auto layer : mynn._layers){
    //    for (int j = 0; j < layer.outputsize; j++){
    //        std::cout << "[ ";
    //        for (int i = 0; i < layer.inputsize; i++)
    //            std::cout << layer.weights[i+j*layer.inputsize];
    //        std::cout << " ][ " << "?" << " ] + [ " << layer.biases[j] << "]" << std::endl;
    //    }
    //}
    auto image = t10k.GetImage(0).Normalize();
    uint32_t label = t10klab.GetLabel(0);
    while (true){
    //for (int i = 0; i < 1000; i++){
        std::cout << "Image " << i << ": Output Neurons: ";
        auto result = mynn(image);
        for (auto e : result)
            std::cout << e << ", ";
        mynn.TrainSingle(image, label);
        std::cout << std::endl;
    //}
    }
}