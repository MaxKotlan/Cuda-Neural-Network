#ifndef IDX_CUDA_H
#define IDX_CUDA_H
#include "idx.h"
#include "image-cuda.h"
#include <thrust/device_vector.h>

namespace IDX{
    class CudaImageDatabase : public ImageDatabase {
        public:
            CudaImageDatabase(std::string filename);
            CudaImage GetImage(unsigned int index);
            
        protected:
            thrust::device_vector<unsigned char> device_raw_data;  

    };
};

#endif

