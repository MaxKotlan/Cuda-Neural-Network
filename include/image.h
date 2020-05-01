#ifndef IMAGE_H
#define IMAGE_H
#include <vector>
#include <assert.h>
#include <climits>
#include <string>
class Image{
    public:
        inline uint32_t x() { return _x;};
        inline uint32_t y() { return _y;};

        inline uint32_t size() {return (_x*_y);}
        inline unsigned char& operator[](unsigned int index) { assert(index < size()); return *(begin + index); }

        std::vector<float> Normalize(){
            std::vector<float> result(_x*_y);
            for (int i = 0; i < size(); i++) 
                result[i] = (float)begin[i] / (float)UCHAR_MAX;
            return std::move(result);
        }

        Image(uint32_t x, uint32_t y, unsigned char* beginaddr) : _x(x), _y(y), begin(beginaddr) { };

    public:
        uint32_t _x;
        uint32_t _y;
        unsigned char* begin;
};

#endif