#include "idx.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include "debug.h"

namespace IDX
{
    Database::Database(std::string filename){
        DEBUG(filename << ": Checking File Headers" << std::endl);
        
    }

    ImageDatabase::ImageDatabase(std::string filename) : Database(filename){
        DEBUG(filename << ": Treating as Image Database" << std::endl);
        assert(type      == DataType::unsignedbyte_);
        assert(dimension == 3);

    }


}