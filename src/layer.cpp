#include "layer.h"

LayerConnector::LayerConnector(uint32_t inputsize, uint32_t outputsize):
    inputsize(inputsize),
    outputsize(outputsize),
    biases(std::move(std::vector<float>(outputsize))), 
    weights(std::move(std::vector<float>(inputsize*outputsize))) 
{
    for (float &bias : biases)
        bias = (float)rand() / (float)RAND_MAX;

    for (float &weight : weights)
        weight = (float)rand() / (float)RAND_MAX;
};

static int wow = 0;

std::vector<float> LayerConnector::operator()(std::vector<float> &previous){
    //previous * weights + bias
    auto result = std::vector<float>(outputsize);
    for (auto &e : result)
        e = wow++;
    return std::move(result);
}