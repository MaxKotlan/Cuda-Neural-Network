#ifndef NORMALIZE_BYTE_H
#define NORMALIZE_BYTE_H
#include <cuda.h>

__global__ void NormalizeByte(unsigned char* dev_raw, float* outputraw, unsigned int size);

#endif