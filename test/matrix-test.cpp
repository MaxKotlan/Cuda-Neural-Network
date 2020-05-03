#include "matrix-test.h"
#include <matrix.h>
#include <thrust/device_vector.h>

void TestMatrix(){
    std::cout << "TESTING MATRIX" << std::endl;

    std::vector<float> mat_a{ 0,1,
                              2,3,
                              4,5};
    std::vector<float> mat_b{ 6, 7, 8, 9,
                             10,11,12,13};
    std::vector<float> mat_c(3*4);

    std::vector<float> shouldbe{
        10,11,12,13,
        42,47,52,57,
        74,83,92,101
    };

    thrust::device_vector<float> dev_mat_a(mat_a.begin(), mat_a.end());
    thrust::device_vector<float> dev_mat_b(mat_b.begin(), mat_b.end());
    thrust::device_vector<float> dev_mat_c(mat_c.size());

    MatrixMultiply(3,2,4, 1.0, 0.0, dev_mat_a, dev_mat_b, dev_mat_c);
    thrust::copy(dev_mat_c.begin(), dev_mat_c.end(), mat_c.begin());

    for (int i = 0; i < mat_c.size(); i++)
        assert(shouldbe[i] == mat_c[i]);
    std::cout << "MATRIX IS CORRECT" << std::endl;

}