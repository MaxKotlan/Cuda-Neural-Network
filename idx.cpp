#include "idx.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include "debug.h"

namespace IDX
{
    
    Database::Database(std::string filename) : filename(filename){
        DEBUG(filename << ": Checking File Headers" << std::endl);
        database.open(filename, std::ios::in | std::ios::binary);
        if (database.is_open()){
            unsigned char padding;
            database.read(&padding, 1); assert(padding == 0); //IDX standard dictates first two bytes are zero
            database.read(&padding, 1); assert(padding == 0);
            database.read(&type, 1);
            database.read(&dimension, 1);
        } 
        else { std::cout << filename << ": Could not open. Exiting..." << std::endl; exit(-1); }
    }

    ImageDatabase::ImageDatabase(std::string filename) : Database(filename){
        DEBUG(filename << ": Treating as Image Database" << std::endl);
        assert(type      == DataType::unsignedbyte_);
        assert(dimension == 3);
        image_count = read_u32();
        DEBUG(filename << ": image_count = " << image_count << std::endl);
        image_x     = read_u32();
        DEBUG(filename << ": image_x = " << image_x << std::endl);
        image_y     = read_u32();
        DEBUG(filename << ": image_y = " << image_y << std::endl);
        CopyRawData(image_count*image_x*image_y);
    }

    std::vector<unsigned char> ImageDatabase::GetImage(unsigned int index){
        unsigned int offset = index*image_x*image_y;
        return std::move(std::vector<unsigned char>(raw_data.begin()+offset, raw_data.begin()+offset+image_x*image_y));
    }


    uint32_t Database::read_u32(){
        uint32_t result;
        unsigned char temp[sizeof(uint32_t)];
        database.read(temp, sizeof(uint32_t));
        result = temp[3] | (temp[2] << 8) | (temp[1] << 16) | (temp[0] << 24);
        return result;
    }

    void Database::CopyRawData(unsigned int bytes){
        DEBUG(filename << ": Copying Raw Data Into Memory" << std::endl);
        raw_data.reserve(bytes);
        database.read(raw_data.data(), bytes);
        DEBUG(filename << ": Finished Copying Raw Data Into Memory" << std::endl);
    }


}