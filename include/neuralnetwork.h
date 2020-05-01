#ifndef NeuralNetwork_H
#define NeuralNetwork_H
#include <stdint.h>
#include <vector>
#include "layer.h"

class NeuralNetwork{
    public:

    NeuralNetwork(uint32_t inputsize, uint32_t hiddenlayersize, uint32_t hiddenlayercount, uint32_t outputsize, float learningrate=1.0);

    /*Forward Propogate Data*/
    thrust::device_vector<float> operator()(thrust::device_vector<float>& input);
    thrust::device_vector<float> ForwardPropagate(thrust::device_vector<float>& input);
    std::vector<float> operator() (std::vector<float>& input);
    std::vector<float> ForwardPropagate(std::vector<float>& input);

    /*Cost of a single example*/
    void TrainSingle(std::vector<float>& input, uint32_t correct);
    void TrainSingle(thrust::device_vector<float>& input, uint32_t correct);
    void Reset();

    inline float getLearningRate()  { return  _learning_rate;};
    inline uint32_t getTrainingCount() { return _training_count;};

    private:
        float _learning_rate;
        uint32_t _training_count; 
        uint32_t _inputsize;
        uint32_t _outputsize;
        std::vector<LayerConnector> _layers;
};

#endif