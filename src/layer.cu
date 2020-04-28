#include "layer.h"
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include "debug.h"

LayerConnector::LayerConnector(uint32_t inputsize, uint32_t outputsize):
    inputsize(inputsize),
    outputsize(outputsize),
    biases(std::move(std::vector<float>(outputsize))), 
    weights(std::move(std::vector<float>(inputsize*outputsize))) 
{
    InitalizeWithRandomValues();
};

void LayerConnector::InitalizeWithRandomValues(){
    float max_range=10.0f;
    for (float &bias : biases)
        bias = max_range*((float)rand() / (float)RAND_MAX)-max_range/2.0f;

    for (float &weight : weights)
        weight = max_range*((float)rand() / (float)RAND_MAX)-max_range/2.0f;// / (float)RAND_MAX;
}

void testssgem(){

    std::vector<float> a{1.,2.,
                         3.,4.,
                         5.,6.};
    std::vector<float> b{7.,8.,9.,10.,
                         11.,12.,13.,14.};
    std::vector<float> c{1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,};

    thrust::device_vector<float> d_a(a.size());
    thrust::device_vector<float> d_b(b.size());
    thrust::device_vector<float> d_c(c.size());

    thrust::copy(a.begin(), a.end(), d_a.begin());
    thrust::copy(b.begin(), b.end(), d_b.begin());
    thrust::copy(c.begin(), c.end(), d_c.begin());

    float alpha = 1.0;
    float beta  = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);

    int m = 3;
    int k = 2;
    int n = 4;

    cublasSgemm(   
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        n, m, k, 
        &alpha,
        thrust::raw_pointer_cast(d_b.data()), n,
        thrust::raw_pointer_cast(d_a.data()), k,
        &beta,
        thrust::raw_pointer_cast(d_c.data()), n
    );

    cublasDestroy(handle);
    thrust::copy(d_a.begin(), d_a.end(), a.begin());
    thrust::copy(d_b.begin(), d_b.end(), b.begin());
    thrust::copy(d_c.begin(), d_c.end(), c.begin());
    
    std::cout << "-----------------------" << std::endl;

    for (auto e : a)
        std::cout << e << " ";
    std::cout << std::endl << std::endl;

    for (auto e : b)
        std::cout << e << " ";
    std::cout << std::endl << std::endl;

    for (auto e : c)
        std::cout << e << " ";
    std::cout << std::endl << std::endl;

}

std::vector<float> LayerConnector::operator()(std::vector<float> &previous){
    //previous * weights + bias
    //testssgem();
    DEBUG("LayerConnector: Input - "); for (auto e : previous) DEBUG(e << ", "); DEBUG(std::endl); 
    auto result = CalculateOutputNeurons(previous);
    DEBUG("LayerConnector: Output - "); for (auto e : result) DEBUG(e << ", "); DEBUG(std::endl); 
    return std::move(result);
}

#include "enable_cuda.h"
#ifdef CUDA_ENABLE

std::vector<float> LayerConnector::CalculateOutputNeurons(std::vector<float>& input){    
    cublasHandle_t handle;
    cublasCreate(&handle);

    thrust::device_vector<float> d_input(input.size());
    thrust::device_vector<float> d_weights(weights.size());
    thrust::device_vector<float> d_output(biases.size());

    thrust::copy(input.begin(),   input.end(),   d_input.begin());
    thrust::copy(weights.begin(), weights.end(), d_weights.begin());
    thrust::copy(biases.begin(),  biases.end(),  d_output.begin());

    int m = d_output.size();
    int k = input.size();
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
    std::vector<float> result(biases.size());
    thrust::copy(d_output.begin(), d_output.end(), result.begin());
    return std::move(result);
}

#else

std::vector<float> LayerConnector::CudaCalculateNeurons(std::vector<float>& input){
    return std::move(std::vector<float>(outputsize));
}


#endif