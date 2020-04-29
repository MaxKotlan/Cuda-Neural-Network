#include "layer.h"
#include <cublas_v2.h>
#include "debug.h"
#include "activation.h"

LayerConnector::LayerConnector(uint32_t inputsize, uint32_t outputsize):
    inputsize(inputsize),
    outputsize(outputsize),
    biases(outputsize), 
    weights(inputsize*outputsize),
    d_input(inputsize),
    d_weights(inputsize*outputsize),
    d_biases(outputsize)
{
    InitalizeWithRandomValues();
    thrust::copy(weights.begin(), weights.end(), d_weights.begin());
    thrust::copy(biases.begin(),  biases.end(),  d_biases.begin());
};

void LayerConnector::InitalizeWithRandomValues(){
    float max_range=10.0f;
    for (float &bias : biases)
        bias = max_range*((float)rand() / (float)RAND_MAX)-max_range/2.0f;

    for (float &weight : weights)
        weight = max_range*((float)rand() / (float)RAND_MAX)-max_range/2.0f;// / (float)RAND_MAX;
}

thrust::device_vector<float> LayerConnector::operator()(thrust::device_vector<float> &d_input){
    //previous * weights + bias
    //testssgem();
    auto result = CalculateOutputNeurons(d_input);
    return std::move(result);
}

thrust::device_vector<float> LayerConnector::CalculateOutputNeurons(thrust::device_vector<float>& d_input){    
    cublasHandle_t handle;
    cublasCreate(&handle);

    thrust::device_vector<float> d_output(outputsize);
    thrust::copy(d_biases.begin(), d_biases.end(), d_output.begin());

    int m = d_output.size();
    int k = d_input.size();
    int n = 1;
    float alpha = 1.0;
    float beta  = 1.0; //1.0 because bias vector added and used as output vector
    cublasSgemm(   
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        n, m, k, 
        &alpha,
        thrust::raw_pointer_cast(d_input.data())  , n,
        thrust::raw_pointer_cast(d_weights.data()), k,
        &beta,
        thrust::raw_pointer_cast(d_output.data()) , n
    );
    cublasDestroy(handle);
    thrust::transform(d_output.begin(), d_output.end(), d_output.begin(), Activation::Sigmoid());
    return std::move(d_output);
}