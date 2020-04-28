#ifndef LAYER_H
#define LAYER_H
#include <stdint.h>
#include <vector>
#include <iostream>
#include <thrust/device_vector.h>

class LayerConnector{

    public:
        LayerConnector(){};
        LayerConnector(uint32_t inputsize, uint32_t outputsize);

        void InitalizeWithRandomValues();
        thrust::device_vector<float> CalculateOutputNeurons(thrust::device_vector<float>& input);
        thrust::device_vector<float> operator() (thrust::device_vector<float>& neurons);

        int size(){
            std::cout << inputsize << "->" << outputsize << " : ";
            return 0;
        }

    public:
        uint32_t inputsize;
        uint32_t outputsize;
        std::vector<float> weights;
        std::vector<float> biases;
        thrust::device_vector<float> d_input;
        thrust::device_vector<float> d_weights;
        thrust::device_vector<float> d_biases;
};

#endif