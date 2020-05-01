#include "idx-cuda.h"
#include "debug.h"

namespace IDX
{
    CudaImageDatabase::CudaImageDatabase(std::string filename) : ImageDatabase(filename)
    {
        DEBUG("Copying Raw Data Into VRAM" << std::endl);
        device_raw_data = raw_data;
        DEBUG("Done Copying Raw Data Into VRAM" << std::endl);
    }

    CudaImage CudaImageDatabase::GetImage(unsigned int index){
        assert(index < image_count && index >= 0);
        unsigned int offset = index*image_x*image_y;
        CudaImage result(image_x, image_y, thrust::raw_pointer_cast(device_raw_data.data()));
        return result;
    }
}