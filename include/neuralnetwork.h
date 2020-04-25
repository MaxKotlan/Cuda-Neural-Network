#ifndef NeuralNetwork_H
#define NeuralNetwork_H
#include <stdint.h>
#include <vector>

class Layer{

    Layer(uint32_t hiddenlayersize) : biases(hiddenlayersize) {};

    private:
        std::vector<float> weights;
        std::vector<float> biases; 
};

class NeuralNetwork{
    public:

    NeuralNetwork(uint32_t inputsize, uint32_t hiddenlayersize, uint32_t hiddenlayercount, uint32_t outputsize );

    private:
        uint32_t _inputsize;
        uint32_t _outputsize;
        std::vector<Layer> _layers;
};

#endif