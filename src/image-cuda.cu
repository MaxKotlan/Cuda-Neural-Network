#include "image-cuda.h"
#include "normalizebyte.h"

thrust::device_vector<float> CudaImage::Normalize(){
    thrust::device_vector<float> result(_x*_y);
    NormalizeByte<<< _x*_y / 512 + 1, 512 >>>(begin, thrust::raw_pointer_cast(result.data()), _x*_y);
    return std::move(result);
}