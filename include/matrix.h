#ifndef MATRIX_H
#define MATRIX_H
#include <thrust/device_vector.h>

void MatrixMultiply(
    uint32_t m, uint32_t k, uint32_t n,
    float alpha, float beta,
    thrust::device_vector<float>& mat_a, 
    thrust::device_vector<float>& mat_b,
    thrust::device_vector<float>& mat_c
);

#endif