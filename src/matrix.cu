#include "matrix.h"
#include <cublas_v2.h>

/*Tricks Cublas into Performing Row Major Order Matrix Multiplication using Matrix Transposes*/
void MatrixMultiply(
    int m, int k, int n,
    float alpha, float beta,
    thrust::device_vector<float>& mat_a, 
    thrust::device_vector<float>& mat_b,
    thrust::device_vector<float>& mat_c
){
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(   
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        n, m, k, 
        &alpha,
        thrust::raw_pointer_cast(mat_b.data()), n,
        thrust::raw_pointer_cast(mat_a.data()), k,
        &beta,
        thrust::raw_pointer_cast(mat_c.data()), n
    );
    cublasDestroy(handle);
}