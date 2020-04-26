#ifndef LAYER_H
#define LAYER_H
#include <stdint.h>
#include <vector>
#include <iostream>

class LayerConnector{

    public:
        LayerConnector(){};
        LayerConnector(uint32_t inputsize, uint32_t outputsize);

        void InitalizeWithRandomValues();

        std::vector<float> operator() (std::vector<float>& neurons);

        int size(){
            std::cout << inputsize << "->" << outputsize << " : ";
            return 0;
        }

    public:
        uint32_t inputsize;
        uint32_t outputsize;
        std::vector<float> weights;
        std::vector<float> biases; 
};

#endif