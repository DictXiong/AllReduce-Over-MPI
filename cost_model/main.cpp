#include <iostream>
using namespace std;
#include "ChooseWidth.h"
#include "CostModel.h"
#include "timer.h"

int main()
{
//    int Total_nodes = 840 * 7 * 7;
//    double Chunk_size = 100;
//    auto timer = newplan::Timer();
//    timer.Start();
//    vector<vector<int>> tree = getWidth(Total_nodes);
//    CostModel(tree,Total_nodes,Chunk_size);
//    timer.Stop();
//    cout << "overhead of cost_model: " << timer.MicroSeconds() << " us" << endl;
//    cout << "num of structure: " << tree.size() << endl;
    freopen("numofstru.csv", "w", stdout);
    for(int i = 1; i < 1000; i++) {
        auto timer = newplan::Timer();
        timer.Start();
        auto tmp = getWidth(i);
        CostModel(tmp,i,100);
        timer.Stop();
        cout << tmp.size() << "," << timer.MicroSeconds() << endl;
    }

}