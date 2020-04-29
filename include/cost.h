#ifndef COST_H
#define COST_H
#include <cuda_runtime.h>

namespace Cost{

    class DifferenceSquared {
        public:

        __device__ inline float operator()(float input) const {
            return (1 / (1 + exp(-input)));
        }
    };
    
}

#endif