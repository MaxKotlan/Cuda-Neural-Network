#include "neuralnetwork.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


NeuralNetwork::NeuralNetwork(uint32_t inputsize, uint32_t hiddenlayersize, uint32_t hiddenlayercount, uint32_t outputsize) 
: _inputsize(inputsize), _outputsize(outputsize),
_layers(std::move(std::vector<LayerConnector>(2+hiddenlayercount)))
{
    _layers[0] = std::move(LayerConnector(inputsize, hiddenlayersize));
    for (int i = 0; i <= hiddenlayercount; i++){
        _layers[i+1] = std::move(LayerConnector(hiddenlayersize, hiddenlayersize));
    }
    _layers[hiddenlayercount+1] = std::move(LayerConnector(hiddenlayersize, outputsize));

}

std::vector<float> NeuralNetwork::operator() (std::vector<float>& in){
    auto input = _layers[0](in);
    for (int i = 1; i < _layers.size(); i++)
        input = _layers[i](input);
    return std::move(input);
}
