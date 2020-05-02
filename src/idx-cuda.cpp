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

    CudaImage CudaImageDatabase::GetImage(uint32_t index){
        assert(index < image_count);
        uint32_t offset = index*image_x*image_y;
        CudaImage result(image_x, image_y, thrust::raw_pointer_cast(device_raw_data.data()+offset));
        return result;
    }
}