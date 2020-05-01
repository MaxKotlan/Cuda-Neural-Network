#include "image-cuda.h"
#include "normalizebyte.h"

thrust::device_vector<float> CudaImage::Normalize(){
    thrust::device_vector<float> result(_x*_y);
    NormalizeByte<<< _x*_y / 512 + 1, 512 >>>(begin, thrust::raw_pointer_cast(result.data()), _x*_y);
    cudaDeviceSynchronize();
    return std::move(result);
}

std::vector<unsigned char> CudaImage::toHostVector(){
    std::vector<unsigned char> result(_x*_y);
    thrust::copy(begin, begin+_x*_y, result.begin());
    std::cout << result.size();
    return std::move(result);
}
