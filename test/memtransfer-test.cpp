#include "memtransfer-test.h"
#include <iostream>
#include "memtransfer.h"

void TestMemTransfer(){
    std::cout << "TESTING MEMTRANSFER" << std::endl;
    std::vector<float> test{0,1,2,3,4,5,6,7,8,9,10}
    auto device_vec = ToDevice(test);
    auto compare = ToHost(device_vec);
    std::equal(compare.begin(), compare.end(), test.begin());
}