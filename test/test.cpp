#include <iostream>
#include "database-test.h"
#include "matrix-test.h"
#include "memtransfer-test.h"

int main(int argc, char** argv){
    std::cout << "----STARTING TESTS----" << std::endl;
    TestMatrix();
    TestImageDatabase();
    TestMemTransfer();
    std::cout << "-----ENDING TESTS-----" << std::endl;

}