// g++ -o test test.cpp && ./test

#include<iostream>
#include "candidates.hpp"
#include "timer.h"

void single_candi_main()
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

int main()
{
    newplan::Timer timer;
    size_t thresh = 9;
    size_t max_time = 0, max_time_num = 0;
    for (int i = 60470; i <= 60490; i++)
    {
        timer.Start();
        Candidates can(i, thresh);
        timer.Stop();
        if (max_time < can.candidates.size())
        {
            max_time = can.candidates.size();
            max_time_num = i;
        }
        //if (i % 10000 == 0)
        std::cout << i << " " << can.candidates.size() << " " << timer.MilliSeconds() << std::endl;
    }
    std::cout << "MAX: " << max_time << " when num = " << max_time_num << std::endl;
/*
0 0 0
10000 764 0
20000 1220 1
30000 1892 2
40000 1948 3
50000 1726 4
60000 3249 5
MAX: 6444 when num = 60480
*/
}