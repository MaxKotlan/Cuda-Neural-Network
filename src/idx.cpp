#include "idx.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include "debug.h"


namespace IDX
{
    
    Database::Database(std::string filename) : filename(filename){
        DEBUG(filename << ": Checking File Headers" << std::endl);
        database = fopen(filename.c_str(), "r");
        if (database){
            unsigned char padding;
            fread(&padding, 1, 1, database); assert(padding == 0); //IDX standard dictates first two bytes are zero
            fread(&padding, 1, 1, database); assert(padding == 0);
            fread(&type, 1, 1, database);
            fread(&dimension, 1, 1, database);
        } 
        else { std::cout << filename << ": Could not open " << filename << ". Exiting..." << std::endl; exit(-1); }
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
        fclose(database);
    }


    LabelDatabase::LabelDatabase(std::string filename) : Database(filename){
        DEBUG(filename << ": Treating as Image Database" << std::endl);
        assert(type      == DataType::unsignedbyte_);
        assert(dimension == 1);
        label_count = read_u32();
        DEBUG(filename << ": image_count = " << label_count << std::endl);
        CopyRawData(label_count);
        fclose(database);
    }

    Image ImageDatabase::GetImage(unsigned int index){
        assert(index < image_count && index >= 0);
        unsigned int offset = index*image_x*image_y;
        Image result(image_x, image_y, &raw_data[offset]);
        return result;
    }

    uint32_t LabelDatabase::GetLabel(unsigned int index){
        unsigned int offset = index;
        return (uint32_t)raw_data[offset];
    }

    uint32_t Database::read_u32(){
        uint32_t result;
        unsigned char temp[sizeof(uint32_t)];
        fread(&temp, sizeof(uint32_t), 1, database);
        result = temp[3] | (temp[2] << 8) | (temp[1] << 16) | (temp[0] << 24);
        return result;
    }

    void Database::CopyRawData(unsigned int bytes){
        DEBUG(filename << ": Copying Raw Data Into Memory" << std::endl);
        raw_data.resize(bytes);
        fread(raw_data.data(), bytes, 1, database);
        DEBUG(filename << ": Finished Copying Raw Data (" << raw_data.size() << " BYTES) Into Memory" << std::endl);
    }


}