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
    _neuralnetwork(network)
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
    float max_range=10.0f;
    for (float &bias : biases)
        bias = max_range*((float)rand() / (float)RAND_MAX)-max_range/2.0f;

    for (float &weight : weights)
        weight = max_range*((float)rand() / (float)RAND_MAX)-max_range/2.0f;// / (float)RAND_MAX;
    
    thrust::copy(weights.begin(), weights.end(), d_weights.begin());
    thrust::copy(biases.begin(),  biases.end(),  d_biases.begin());
}

thrust::device_vector<float> LayerConnector::operator()(thrust::device_vector<float> &d_input){
    //previous * weights + bias
    //testssgem();
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
    
    thrust::device_vector<float> d_delta_weight(outputsize*inputsize);
    thrust::device_vector<float> d_delta_bias(outputsize);
    thrust::device_vector<float> d_activation_delta(d_output_ref->size());
    d_activation_delta = *d_output_ref;
    thrust::transform(d_activation_delta.begin(), d_activation_delta.end(), d_activation_delta.begin(), Activation::SigmoidDerivative());
    thrust::transform(d_activation_delta.begin(), d_activation_delta.end(), d_cost.begin(), d_cost.begin(), thrust::multiplies<float>());

    MatrixMultiply(
        d_input.size(), 1, d_cost.size(),
        1.0, 0.0, 
        d_input, d_cost, d_delta_weight
    );

    float learningrate = 1.0;
    thrust::transform(d_delta_weight.begin(), d_delta_weight.end(), thrust::make_constant_iterator(learningrate), d_delta_weight.begin(), thrust::multiplies<float>());
    thrust::transform(d_weights.begin(), d_weights.end(), d_delta_weight.begin(), d_weights.begin(), thrust::minus<float>());

    /*
    thrust::copy(d_delta_weight.begin(), d_delta_weight.end(), weights.begin());
    std::cout << std::endl << "Weights: " << std::endl;
    for (auto e : weights){
        std::cout << e << " ";
    }
    std::cout << std::endl;*/

}
