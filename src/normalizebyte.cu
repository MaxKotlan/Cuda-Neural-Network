#include "normalizebyte.h"

__global__ void NormalizeByte(unsigned char* dev_raw, float* outputraw, unsigned int size){
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        outputraw[idx] = (float)dev_raw[idx] / (float)UCHAR_MAX;
    }
};
