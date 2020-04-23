#include "idx.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include "debug.h"

namespace IDX
{
    
    Database::Database(std::string filename){
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
        DEBUG(filename << ": image_x = " << image_y << std::endl);
    }

    uint32_t Database::read_u32(){
        uint32_t result;
        unsigned char temp[sizeof(uint32_t)];
        database.read(temp, sizeof(uint32_t));
        result = temp[3] | (temp[2] << 8) | (temp[1] << 16) | (temp[0] << 24);
        return result;
    }


}