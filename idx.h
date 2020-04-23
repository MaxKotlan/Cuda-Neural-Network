#ifndef IDX_H
#define IDX_H
#include <vector>
#include <fstream>
#include <string>

namespace IDX{

    enum DataType{
        unsignedbyte_=0x08,
          signedbyte_=0x09,
               short_=0x0B,
                 int_=0x0C,
               float_=0x0D,
              double_=0x0E,
    };

    class Database{

        public:

            Database(std::string filename);

        protected:
            std::ifstream database;
            std::string filename;
            unsigned char type;
            unsigned char dimension;
            std::vector<unsigned char> raw_data;  
    };

    class ImageDatabase : Database{
        
        public:

        ImageDatabase(std::string filename);

        protected:
            int image_count;
            int image_x;
            int image_y;
            unsigned int start_offset;
    };

}

#endif 