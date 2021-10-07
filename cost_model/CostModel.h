double latency_control_overhead(double Chunk_size, double Tree_width)
{
    double lo  = 0.004;
    double co = 0.0002;
    double tw = Tree_width;
    double s = Chunk_size;
    if(tw > 9)
    {
        double cost = 2 * lo
                      + s * (tw - 9) * co;
        cout << "the latency & control overhead of the layer is: " << cost << endl;
        return cost;
    }
    else
    {
        double cost = 2 * lo;
        cout << "the latency & control overhead of the layer is: " << cost << endl;
        return cost;
    }
}

double bandwidth_calculation_overhead(int Total_nodes, double Chunk_size)
{
    double bo = 0.0068;
    double s = Chunk_size;
    double n = Total_nodes;
    double cost = (((n - 1) / n) * s) * bo;
    cout << "the overhead of the bandwidth & calculation part is: " << cost << endl;
    return cost;
}

double memory_read_write_overhead(vector<int> tree_structure, int Tree_height, int Total_nodes, double Chunk_size)
{
    vector<int> tree = tree_structure;
    int th = Tree_height;
    double s = Chunk_size;
    double o = 0.0004;
    int steps;
    int n = Total_nodes;
    switch (th)
    {
        case 1:
            steps = n + 1;
            cout << "the overhead of the memory w / r part is: "<< ((steps * s) / n) * o << endl;
            return ((steps * s) / n) * o;
        case 2:
            steps = n + 2 * tree[0] + 1;
            cout << "the overhead of the memory w / r part is: "<< ((steps * s) / n) * o << endl;
            return ((steps * s) / n) * o;
        case 3:
            steps = n + 2 * tree[0] * tree[1] + 2 * tree[0] + 1;
            cout << "the overhead of the memory w / r part is: "<< ((steps * s) / n) * o << endl;
            return ((steps * s) / n) * o;
        case 4:
            steps = n + 2 * tree[0] * tree[1] * tree[2] + 2 * tree[0] * tree[1] + 2 * tree[0] + 1;
            cout << "the overhead of the memory w / r part is: "<< ((steps * s) / n) * o << endl;
            return ((steps * s) / n) * o;
        case 5:
            steps = n + 2 * tree[0] * tree[1] * tree[2] * tree[3] + 2 * tree[0] * tree[1] * tree[2] + 2 * tree[0] * tree[1] + 2 * tree[0] + 1;
            cout << "the overhead of the memory w / r part is: "<< ((steps * s) / n) * o << endl;
            return ((steps * s) / n) * o;
        case 6:
            steps = n + 2 * tree[0] * tree[1] * tree[2] * tree[3] * tree[4] + 2 * tree[0] * tree[1] * tree[2] * tree[3] + 2 * tree[0] * tree[1] * tree[2] + 2 * tree[0] * tree[1] + 2 * tree[0] + 1;
            cout << "the overhead of the memory w / r part is: "<< ((steps * s) / n) * o << endl;
            return ((steps * s) / n) * o;
        case 7:
            steps = n + 2 * tree[0] * tree[1] * tree[2] * tree[3] * tree[4] * tree[5] + 2 * tree[0] * tree[1] * tree[2] * tree[3] * tree[4] + 2 * tree[0] * tree[1] * tree[2] * tree[3] + 2 * tree[0] * tree[1] * tree[2] + 2 * tree[0] * tree[1] + 2 * tree[0] + 1;
            cout << "the overhead of the memory w / r part is: "<< ((steps * s) / n) * o << endl;
            return ((steps * s) / n) * o;
        case 8:
            steps = n + 2 * tree[0] * tree[1] * tree[2] * tree[3] * tree[4] * tree[5] * tree[6] + 2 * tree[0] * tree[1] * tree[2] * tree[3] * tree[4] * tree[5] + 2 * tree[0] * tree[1] * tree[2] * tree[3] * tree[4] + 2 * tree[0] * tree[1] * tree[2] * tree[3] + 2 * tree[0] * tree[1] * tree[2] + 2 * tree[0] * tree[1] + 2 * tree[0] + 1;
            cout << "the overhead of the memory w / r part is: "<< ((steps * s) / n) * o << endl;
            return ((steps * s) / n) * o;
        case 9:
            steps = n + 2 * tree[0] * tree[1] * tree[2] * tree[3] * tree[4] * tree[5] * tree[6] * tree[7] + 2 * tree[0] * tree[1] * tree[2] * tree[3] * tree[4] * tree[5] * tree[6] + 2 * tree[0] * tree[1] * tree[2] * tree[3] * tree[4] * tree[5] + 2 * tree[0] * tree[1] * tree[2] * tree[3] * tree[4] + 2 * tree[0] * tree[1] * tree[2] * tree[3] + 2 * tree[0] * tree[1] * tree[2] + 2 * tree[0] * tree[1] + 2 * tree[0] + 1;
            cout << "the overhead of the memory w / r part is: "<< ((steps * s) / n) * o << endl;
            return ((steps * s) / n) * o;
    }
}


void CostModel(vector<vector<int>> tree, int Total_nodes, double Chunk_size)
{
    double cost_output = 1000000000;
    int output_index;
    for(int i = 0; i < tree.size(); i++)
    {
        cout << "*------------start analysing one single structure---------*" << endl;
        double cost;
        for (int j = 0; j < tree[i].size(); j++) {
            cout << "the width of the layer is: " << tree[i][j] << endl;
            cost += latency_control_overhead(100, tree[i][j]);
        }
        cost += memory_read_write_overhead(tree[i], tree[i].size(), Total_nodes, Chunk_size);
        cost += bandwidth_calculation_overhead(Total_nodes, Chunk_size);
        cout << "the single cost should be: " << cost << endl;
        if(cost < cost_output)
        {
            cost_output = cost;
            output_index = i;
        }
        cost = 0;
        cout << "*-----------finish analysing one single structure---------*" << endl;
        cout << endl;
    }
    cout << "the optimized tree structure for " << Total_nodes << " total nodes should be: ";
    for(int i = 0; i < tree[output_index].size(); i++)
    {
        if(i == tree[output_index].size()-1)
        {
            cout << tree[output_index][i];
        }
        else
        {
            cout << tree[output_index][i] << "*";
        }
    }
    cout << endl;
    cout << "the optimized all_reduce time for " << Total_nodes << " total nodes should be: " << cost_output << endl;
}