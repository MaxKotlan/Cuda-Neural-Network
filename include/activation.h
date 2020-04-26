#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <tgmath.h>

namespace ACTIVATION
{

    inline float Sigmoid(float input){
        return (1 / (1 + exp(-input));
    }

};


#endif