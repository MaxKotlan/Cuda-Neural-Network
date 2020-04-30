#ifndef MATRIX_H
#define MATRIX_H
#include <thrust/device_vector.h>

void MatrixMultiply(
    int m, int k, int n,
    float alpha, float beta,
    thrust::device_vector<float>& mat_a, 
    thrust::device_vector<float>& mat_b,
    thrust::device_vector<float>& mat_c
);

#endif