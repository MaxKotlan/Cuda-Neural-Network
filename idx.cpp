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
            database.read(&type, 1); 
            database.read(&type, 1);
            database.read(&type, 1);
            database.read(&dimension, 1);
        } 
        else { std::cout << filename << ": Could not open. Exiting..." << std::endl; exit(-1); }
    }

    ImageDatabase::ImageDatabase(std::string filename) : Database(filename){
        DEBUG(filename << ": Treating as Image Database" << std::endl);
        assert(type      == DataType::unsignedbyte_);
        assert(dimension == 3);

    }


}