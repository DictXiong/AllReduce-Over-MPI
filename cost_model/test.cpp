// g++ -o test test.cpp && ./test

#include<iostream>
#include "candidates.hpp"
#include "timer.h"

int main()
{
    newplan::Timer timer;
    size_t N = 227;
    size_t thresh = 9;
    timer.Start();
    Candidates can(N, thresh);
    timer.Stop();
    can.print();
    std::cout << std::endl << "GIVEN #nodes: " << N << ", incast: " << thresh << std::endl << "total: " << can.candidates.size() << std::endl << "time: " << timer.MilliSeconds() << std::endl;
}