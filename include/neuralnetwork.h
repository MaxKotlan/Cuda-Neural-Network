#ifndef NeuralNetwork_H
#define NeuralNetwork_H
#include <stdint.h>

class NeuralNetwork{
    public:

    NeuralNetwork(uint32_t inputsize, uint32_t hiddenlayersize, uint32_t hiddenlayercount, uint32_t outputsize );

    private:
        uint32_t _inputsize;
        uint32_t _hiddenlayersize;
        uint32_t _hiddenlayercount;
        uint32_t _outputsize;
};

#endif