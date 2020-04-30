#ifndef LAYER_H
#define LAYER_H
#include <stdint.h>
#include <vector>
#include <iostream>
#include <thrust/device_vector.h>

class NeuralNetwork;

class LayerConnector{

    public:
        LayerConnector(){};
        LayerConnector(uint32_t inputsize, uint32_t outputsize, NeuralNetwork* network);

        void InitalizeWithRandomValues();
        thrust::device_vector<float> CalculateOutputNeurons(thrust::device_vector<float>& input);
        thrust::device_vector<float> operator() (thrust::device_vector<float>& neurons);

        void CalculateGradient(thrust::device_vector<float>& cost);
        inline void SetNextLayerReference(LayerConnector* nextlayer_ref) { _nextLayer = nextlayer_ref; }
        inline void SetOutputReference(thrust::device_vector<float>* refoutput) { d_output_ref = refoutput; }
        inline thrust::device_vector<float>* GetInputReference() { return &d_input; };

    protected:
        uint32_t inputsize;
        uint32_t outputsize;
        std::vector<float> weights;
        std::vector<float> biases;
        thrust::device_vector<float>  d_input;
        thrust::device_vector<float>* d_output_ref; //refrence to output stored in next layer
        thrust::device_vector<float> d_weights;
        thrust::device_vector<float> d_biases;
        LayerConnector* _nextLayer;
        NeuralNetwork*  _neuralnetwork;
};

#endif