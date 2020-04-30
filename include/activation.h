#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <cuda_runtime.h>

namespace Activation{

    class Sigmoid {
        public:

        __device__ inline float operator()(float input) const {
            return (1 / (1 + exp(-input)));
        }
        
    };

    class SigmoidDerivative {
        public:

        __device__ inline float operator()(float input) const {
            return (input*(1-input));
        }

    };
    
}

#endif