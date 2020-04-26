#ifndef NeuralNetwork_H
#define NeuralNetwork_H
#include <stdint.h>
#include <vector>
#include "layer.h"

class NeuralNetwork{
    public:

    NeuralNetwork(uint32_t inputsize, uint32_t hiddenlayersize, uint32_t hiddenlayercount, uint32_t outputsize );

    std::vector<float> operator() (std::vector<float>& input);
    std::vector<float> ComputeOutputNeurons(std::vector<float>& input);

    public:
        uint32_t _inputsize;
        uint32_t _outputsize;
        std::vector<LayerConnector> _layers;
};

#endif