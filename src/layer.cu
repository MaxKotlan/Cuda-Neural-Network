#include "layer.h"
#include <cublas_v2.h>
#include "debug.h"
#include "activation.h"
#include "matrix.h"

LayerConnector::LayerConnector(uint32_t inputsize, uint32_t outputsize, NeuralNetwork* network=nullptr):
    inputsize(inputsize),
    outputsize(outputsize),
    biases(outputsize), 
    weights(inputsize*outputsize),
    d_input(inputsize),
    d_output_ref(nullptr),
    d_weights(inputsize*outputsize),
    d_biases(outputsize),
    _nextLayer(nullptr),
    _neuralnetwork(network),
    d_delta_weights(outputsize*inputsize),
    d_delta_biases(outputsize),
    d_activation_delta(outputsize)
{
    if (network == nullptr){
        std::cerr << " \n\
        ERROR. You must set a reference to the neural network, \
        so that the layer can retrive global parameters such as the learning rate. \
        ";
        exit(-1);
    }
    InitalizeWithRandomValues();
};

void LayerConnector::InitalizeWithRandomValues(){
    float max_range=2.0f;
    for (float &bias : biases)
        bias = max_range*((float)rand() / (float)RAND_MAX)-max_range/2.0f;

    for (float &weight : weights)
        weight = max_range*((float)rand() / (float)RAND_MAX)-max_range/2.0f;// / (float)RAND_MAX;
    
    thrust::copy(weights.begin(), weights.end(), d_weights.begin());
    thrust::copy(biases.begin(),  biases.end(),  d_biases.begin());
}

thrust::device_vector<float> LayerConnector::operator()(thrust::device_vector<float> &d_input){
    auto result = CalculateOutputNeurons(d_input);
    return std::move(result);
}

thrust::device_vector<float> LayerConnector::CalculateOutputNeurons(thrust::device_vector<float>& d_input_new){
    d_input = std::move(d_input_new);

    thrust::device_vector<float> d_output(outputsize);
    thrust::copy(d_biases.begin(), d_biases.end(), d_output.begin());

    MatrixMultiply(
        d_output.size(), d_input.size(), 1,
        1.0,1.0,
        d_weights, d_input, d_output
    );

    thrust::transform(d_output.begin(), d_output.end(), d_output.begin(), Activation::Sigmoid());
    return std::move(d_output);
}

void LayerConnector::CalculateGradient(thrust::device_vector<float>& d_cost){
    if (d_output_ref == nullptr){
        std::cerr << " \n\
        ERROR. You must set a reference to the input of the next layer to calculate the gradient.\n \
        Please do this by calling SetOutputReference( [your layer].getInputReference() )\n \
        before you call CalculateGradient().\n \
        ";
        exit(-1);
    }

    d_activation_delta = GenerateActivationDelta(*d_output_ref);
    if (_nextLayer == nullptr) {
        thrust::transform(d_activation_delta.begin(), d_activation_delta.end(), d_cost.begin(), d_activation_delta.begin(), thrust::multiplies<float>());
    } else {
        thrust::device_vector<float> previous_layer_delta = CalculatePreviousLayerDelta();
        thrust::transform(d_activation_delta.begin(), d_activation_delta.end(), previous_layer_delta.begin(), d_activation_delta.begin(), thrust::multiplies<float>());
    }

    MatrixMultiply(
        d_input.size(), 1, d_activation_delta.size(),
        1.0, 0.0, 
        d_input, d_activation_delta, d_delta_weights
    );

    thrust::transform(d_activation_delta.begin(), d_activation_delta.end(), d_biases.begin(), d_delta_biases.begin(), thrust::multiplies<float>());
}

thrust::device_vector<float> LayerConnector::CalculatePreviousLayerDelta(){
    thrust::device_vector<float> previous_layer_delta(outputsize);

    /*Im not sure if this is correct*/
    MatrixMultiply(
        _nextLayer->inputsize, _nextLayer->outputsize, 1,
        1.0, 0.0, 
        _nextLayer->d_weights, _nextLayer->d_activation_delta, previous_layer_delta
    );

    /*
    for (int i = 0; i < inputsize; i++){
        thrust::transform(
            _nextLayer->d_weights.begin()+i*inputsize,
            _nextLayer->d_weights.begin()+i*inputsize+outputsize,
            _nextLayer->d_activation_delta, 
            thrust::multiplies<float>()
        );
    }
    thrust*/

    return std::move(previous_layer_delta);
}

thrust::device_vector<float> LayerConnector::GenerateActivationDelta(const thrust::device_vector<float>& output_layer){
    thrust::device_vector<float> d_activation_delta = output_layer;
    thrust::transform(d_activation_delta.begin(), d_activation_delta.end(), d_activation_delta.begin(), Activation::SigmoidDerivative());
    return std::move(d_activation_delta);
}

void LayerConnector::ApplyDeltas(){
    /*Multiply Deltas by the learning rate*/
    float learningrate = 1.0;
    thrust::transform(d_delta_weights.begin(), d_delta_weights.end(), thrust::make_constant_iterator(learningrate), d_delta_weights.begin(), thrust::multiplies<float>());
    thrust::transform(d_delta_biases.begin(), d_delta_biases.end(), thrust::make_constant_iterator(learningrate), d_delta_biases.begin(), thrust::multiplies<float>());

    /*Apply deltas to weights and biases*/
    thrust::transform(d_weights.begin(), d_weights.end(), d_delta_weights.begin(), d_weights.begin(), thrust::minus<float>());
    thrust::transform(d_biases.begin(), d_biases.end(), d_delta_biases.begin(), d_biases.begin(), thrust::minus<float>());
}