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
    NeuralNetwork mynn(28*28, 16, 2, 10, 0.1);
    std::cout << std::fixed << std::setprecision(2);

    uint32_t imageindex;
    std::cout << " Enter Image Index: ";
    std::cin >> imageindex;

    uint32_t pollingrate = 15;
    uint32_t count = 0;
    auto image = t10k.GetImage(imageindex).Normalize();
    uint32_t label = t10klab.GetLabel(imageindex);    
    while (true){

        if (count%pollingrate == 0){
            auto device_result = mynn.ForwardPropagate(image);
            std::vector<float> result(device_result.size());
            thrust::copy(device_result.begin(), device_result.end(), result.begin());
            std::cout << "training iteration: " << count << " should be: " << label << " Probabilities: ";
            for (auto e : result)
                std::cout << e << " ";
            std::cout << std::endl;
        }

        mynn.TrainSingle(image, label);
        count++;
    }
}