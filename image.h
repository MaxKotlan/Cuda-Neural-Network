#ifndef IMAGE_H
#define IMAGE_H
#include <vector>
#include <assert.h>
#include <iostream>
#include <string>
class Image{
    public:
        inline uint32_t x() { return _x;};
        inline uint32_t y() { return _y;};

        unsigned int size() {return (_x*_y);}
        inline unsigned char& operator[](unsigned int index) { assert(index < size()); return *(begin + index); }

        Image(uint32_t x, uint32_t y, unsigned char* beginaddr) : _x(x), _y(y), begin(beginaddr) { };

    private:
        uint32_t _x;
        uint32_t _y;
        unsigned char* begin;
};

#endif