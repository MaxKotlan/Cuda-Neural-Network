#ifndef IMAGE_CUDA_H
#define IMAGE_CUDA_H
#include "image.h"
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

class CudaImage : public Image {

    public:

    thrust::device_vector<unsigned char> Normalize(){
        thrust::device_vector<unsigned char> result(begin, begin+_x*_y);
        //thrust::transform(result.begin(), result.end(), thrust::make_constant_iterator((float)UCHAR_MAX), result.begin(), thrust::divides<float>());
        return result;
    }

    CudaImage(uint32_t x, uint32_t y, unsigned char* beginaddr) : Image(x, y, beginaddr){}
};

#endif