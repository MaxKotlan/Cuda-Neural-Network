#include "layer.h"
#include <cublas_v2.h>
#include <thrust/device_vector.h>

LayerConnector::LayerConnector(uint32_t inputsize, uint32_t outputsize):
    inputsize(inputsize),
    outputsize(outputsize),
    biases(std::move(std::vector<float>(outputsize))), 
    weights(std::move(std::vector<float>(inputsize*outputsize))) 
{
    InitalizeWithRandomValues();
};

void LayerConnector::InitalizeWithRandomValues(){
    for (float &bias : biases)
        bias = (float)rand() / (float)RAND_MAX;

    for (float &weight : weights)
        weight = (float)rand() / (float)RAND_MAX;
}

std::vector<float> LayerConnector::operator()(std::vector<float> &previous){
    //previous * weights + bias
    return std::move(CalculateOutputNeurons(previous));
}

#include "enable_cuda.h"
#ifdef CUDA_ENABLE

std::vector<float> LayerConnector::CalculateOutputNeurons(std::vector<float>& input){
    std::vector<float> result(outputsize);
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    thrust::device_vector<float> d_input(input.size());
    thrust::device_vector<float> d_weights(weights.size());
    thrust::device_vector<float> d_biases(biases.size());
    thrust::device_vector<float> d_output(result.size());

    thrust::copy(weights.begin(), weights.end(), d_weights.begin());
    thrust::copy(biases.begin(), biases.end(), d_biases.begin());

    float alpha = 1.0;
    float beta  = 1.0;
    cublasSgemm(   
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        outputsize, inputsize, 1, 
        &alpha,
        thrust::raw_pointer_cast(d_weights.data()), d_weights.size(),
        thrust::raw_pointer_cast(d_input.data())  , d_input.size(),
        &beta,
        thrust::raw_pointer_cast(d_biases.data()) , d_biases.size()
    );

    cublasDestroy(handle);
    thrust::copy(d_output.begin(), d_output.end(), result.begin());

    return std::move(result);
}

#else

std::vector<float> LayerConnector::CudaCalculateNeurons(std::vector<float>& input){
    return std::move(std::vector<float>(outputsize));
}


#endif