#include <iostream>
#include "database-test.h"
#include "matrix-test.h"

int main(int argc, char** argv){
    std::cout << "STARTING TESTS" << std::endl;
    TestMatrix();
    TestImageDatabase();
}