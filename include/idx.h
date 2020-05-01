#ifndef IDX_H
#define IDX_H
#include <vector>
#include <fstream>
#include <string>
#include "image.h"

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
            uint32_t read_u32();
            void CopyRawData(unsigned int bytes);

        protected:
            FILE* database;
            std::string filename;
            unsigned char type;
            unsigned char dimension;
            std::vector<unsigned char> raw_data;  
    };

    class ImageDatabase : protected Database{
        
        public:

        ImageDatabase(std::string filename);
        Image GetImage(unsigned int index);

        inline uint32_t size() { return image_count; } 
        inline uint32_t x()    { return image_x; } 
        inline uint32_t y()    { return image_y; } 

        protected:
            uint32_t image_count;
            uint32_t image_x;
            uint32_t image_y;
    };

    class LabelDatabase : protected Database {
        public:

        LabelDatabase(std::string filename);
        uint32_t GetLabel(unsigned int index);

        protected:
            uint32_t label_count;

    };

}

#endif 