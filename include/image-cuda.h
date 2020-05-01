#ifndef IMAGE_CUDA_H
#define IMAGE_CUDA_H
#include "image.h"
#include <thrust/device_vector.h>

class CudaImage : protected Image {

    public:

    thrust::device_vector<float> Normalize();
    std::vector<unsigned char> toHostVector();

    CudaImage(uint32_t x, uint32_t y, unsigned char* beginaddr) : Image(x, y, beginaddr){}
};

#endif