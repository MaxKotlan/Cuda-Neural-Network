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
    _layers[0] = LayerConnector(inputsize, hiddenlayersize, this);
    for (int i = 1; i < hiddenlayercount; i++){
        DEBUG("LAYERCONNECTOR " <<i<<": Initilizing layer connector " << hiddenlayersize << " => " << hiddenlayersize << std::endl);
        _layers[i] = LayerConnector(hiddenlayersize, hiddenlayersize, this);
    }
    DEBUG("LAYERCONNECTOR " <<hiddenlayercount<<": Initilizing layer connector " << hiddenlayersize << " => " << outputsize << std::endl);
    _layers[hiddenlayercount] = LayerConnector(hiddenlayersize, outputsize, this);

    /*Set references to next layer/output*/
    for (int i = 0; i < _layers.size()-1; i++){
        _layers[i].SetNextLayerReference(&_layers[i+1]);
        _layers[i].SetOutputReference(_layers[i+1].GetInputReference());
    }

}

std::vector<float> NeuralNetwork::operator() (std::vector<float>& in){
    return std::move(ForwardPropagate(in));
}

thrust::device_vector<float> NeuralNetwork::DeviceForwardPropagate(std::vector<float>& input){
    thrust::device_vector<float> d_input = input;
    for (auto &layer : _layers)
        d_input = layer(d_input);
    return std::move(d_input);
}


std::vector<float> NeuralNetwork::ForwardPropagate(std::vector<float>& input){
    thrust::device_vector<float> d_input = input;
    d_input = DeviceForwardPropagate(input);
    std::vector<float> outvec(d_input.size());
    thrust::copy(d_input.begin(), d_input.end(), outvec.begin());
    return std::move(outvec);
}

void NeuralNetwork::TrainSingle(std::vector<float>& input, uint32_t correct){
    auto outputlayer = DeviceForwardPropagate(input);

    /*Definetly making a custom transform for this*/

    std::cout << "Correct: " << correct;

    float correctvalue = 1.0;
    float incorrectvalue = 0.0;

    thrust::device_vector<float> cost(outputlayer.size());
    thrust::fill(cost.begin(), cost.end(), incorrectvalue);
    thrust::copy(&correctvalue, &correctvalue+1, (cost.begin()+correct));
    thrust::transform(outputlayer.begin(), outputlayer.end(), cost.begin(), cost.begin(), thrust::minus<float>());
    thrust::transform(cost.begin(), cost.end(), thrust::make_constant_iterator(2), cost.begin(), thrust::multiplies<float>());
    
    _layers[_layers.size()-1].SetOutputReference(&outputlayer);
    for (int i = _layers.size()-1; i >= 0; i--){
        _layers[i].CalculateGradient(cost);
    } 
}

void NeuralNetwork::Reset(){
    for (auto &layer : _layers)
        layer.InitalizeWithRandomValues();
}