#ifndef MEMTRANSFER_H
#define MEMTRANSFER_H
#include <thrust/device_vector.h>

template <typename T>
thrust::device_vector<T> ToDevice(std::vector<T>& input){
    return std::move(thrust::device_vector<T>(input.begin(), input.end()));
}

template <typename T>
std::vector<T> ToHost(thrust::device_vector<T>& input){
    std::vector<T> result(input.size());
    thrust::copy(input.begin(), input.end(), result.begin());
    return std::move(result);
}

#endif