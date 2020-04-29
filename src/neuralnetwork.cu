#include "neuralnetwork.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "debug.h"

NeuralNetwork::NeuralNetwork(uint32_t inputsize, uint32_t hiddenlayersize, uint32_t hiddenlayercount, uint32_t outputsize) 
: _inputsize(inputsize), _outputsize(outputsize),
_layers(1+hiddenlayercount)
{
    DEBUG("NEURALNETWORK: Initalizing with inputsize:" << inputsize << ", hiddenlayersize: " << hiddenlayersize 
    << ", hiddenlayercount: " << hiddenlayercount << ", outputsize: " << outputsize << std::endl);
    DEBUG("LAYERCONNECTOR 0: Initilizing layer connector " << inputsize << " => " << hiddenlayersize << std::endl);
    _layers[0] = LayerConnector(inputsize, hiddenlayersize);
    for (int i = 1; i < hiddenlayercount; i++){
        DEBUG("LAYERCONNECTOR " <<i<<": Initilizing layer connector " << hiddenlayersize << " => " << hiddenlayersize << std::endl);
        _layers[i] = LayerConnector(hiddenlayersize, hiddenlayersize);
    }
    DEBUG("LAYERCONNECTOR " <<hiddenlayercount<<": Initilizing layer connector " << hiddenlayersize << " => " << outputsize << std::endl);
    _layers[hiddenlayercount] = LayerConnector(hiddenlayersize, outputsize);

}

std::vector<float> NeuralNetwork::operator() (std::vector<float>& in){
    return std::move(ForwardPropogate(in));
}

std::vector<float> NeuralNetwork::ForwardPropogate(std::vector<float>& input){
    thrust::device_vector<float> d_input = input;
    for (int i = 0; i < _layers.size(); i++)
        d_input = _layers[i](d_input);
    std::vector<float> outvec(d_input.size());
    thrust::copy(d_input.begin(), d_input.end(), outvec.begin());
    return std::move(outvec);
}