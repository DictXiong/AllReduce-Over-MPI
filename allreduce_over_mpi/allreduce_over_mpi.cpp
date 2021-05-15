#include<iostream>
#include<vector>
#include "glog/logging.h"

class Operation
{
public:    
    size_t peer;
    std::vector<size_t> blocks;
    /**
     * Operation 类构造函数.
     *  
     * @param _peer 接受/发送操作的对象
     * @param _total_peers 参与计算的总节点数
     * @param _gap 接收/发送的数据块的编号间距
     */
    Operation(size_t _peer, size_t _total_peers, size_t _gap): peer(_peer)
    {
        size_t start = _peer % _gap;
        for (size_t i = start; i < _total_peers; i += _gap)
        {
            blocks.push_back(i);
        }
    }
};

class Operations
{
protected:
    std::vector<size_t> stages;
    size_t total_peers, num_peer;
public:
    std::vector<std::vector<Operation>> ops;
    /**
     * Operations 类的构造函数
     * 
     * @param _total_peers 参与计算的总节点数
     * @param _num_peer 当前节点的编号
     * @param _stages 一个向量, 记录了 AllReduce 树自下而上每一层的宽度. 注意积应当等于 {@code _total_peers}.
     */ 
    Operations(size_t _total_peers, size_t _num_peer, std::vector<size_t> _stages): total_peers(_total_peers), num_peer(_num_peer), stages(_stages)
    {
        CHECK_GT(_total_peers, 0);
        CHECK_GE(_num_peer, 0);
        CHECK_LT(_num_peer, _total_peers);
        size_t pi = 1;
        for (const auto &i:_stages)
        {
            CHECK_GT(i, 0);
            pi *= i;
        }
        CHECK_EQ(pi, _total_peers);
    }
    // 生成拓扑, 要求子类实现
    virtual void generate_ops() = 0;
    // 打印拓扑
    virtual void print_ops()const
    {
        std::cout << typeid(*this).name() << " of node " << num_peer << " in total " << total_peers << " peers: " << std::endl;
        for (const auto &i:ops)
        {
            if (&i != &*(ops.end() - 1))
            {
                std::cout << "┝ stage";
            }
            else 
            {
                std::cout << "┕ stage";
            }
            for (const auto &j:i)
            {
                std::cout<< " | node " << j.peer<<": ";
                for (auto k:j.blocks)
                {
                    std::cout<<k<<",";
                }
            }
            std::cout<<std::endl;
        }
    }
};

class Send_Ops: public Operations
{
public:
    using Operations::Operations;
    // 生成逻辑拓扑
    virtual void generate_ops()
    {
        // 当前组内成员的编号的间距
        size_t gap = 1;
        for (auto i:stages)
        {
            std::vector<Operation> stage_ops;
            // 当前组内编号最小的成员
            size_t left_peer = num_peer / (gap * i) * (gap * i) + num_peer % gap;
            for (size_t j = 0; j < i; j++)
            {
                stage_ops.emplace_back(left_peer, total_peers, gap * i);
                left_peer += gap;
            }
            ops.push_back(stage_ops);
            gap *= i;
        }
    }
};

class Recv_Ops: public Operations
{
public:
    using Operations::Operations;
    // 生成逻辑拓扑
    virtual void generate_ops()
    {
        // 当前组内成员的编号的间距
        size_t gap = 1;
        for (auto i:stages)
        {
            std::vector<Operation> stage_ops;
            Operation op_template(num_peer, total_peers, gap * i);
            // 当前组内编号最小的成员
            size_t left_peer = num_peer / (gap * i) * (gap * i) + num_peer % gap;
            for (size_t j = 0; j < i; j++)
            {
                op_template.peer = left_peer;
                stage_ops.emplace_back(op_template);
                left_peer += gap;
            }
            ops.push_back(stage_ops);
            gap *= i;
        }
    }
};

template<typename T>
void reduce_sum(T **src, int num_blocks, size_t num_elements)
{
#define PARALLEL_THREAD 14
    T *dst = src[0];
    T *src0 = src[0];
    T *src1 = src[1];
    T *src2 = src[2];
    T *src3 = src[3];
    T *src4 = src[4];
    T *src5 = src[5];
    T *src6 = src[6];
    T *src7 = src[7];
    T *src8 = src[8];
    T *src9 = src[9];
    T *src10 = src[10];
    T *src11 = src[11];
    T *src12 = src[12];
    T *src13 = src[13];
    T *src14 = src[14];
    T *src15 = src[15];
    T *src16 = src[16];
    T *src17 = src[17];
    T *src18 = src[18];
    T *src19 = src[19];

    switch (num_blocks)
    {
    case 2:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i];
        }
        break;
    }
    case 3:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i];
        }
        break;
    }
    case 4:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i];
        }
        break;
    }
    case 5:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i];
        }
        break;
    }
    case 6:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i];
        }
        break;
    }
    case 7:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i];
        }
        break;
    }
    case 8:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i];
        }
        break;
    }
    case 9:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i];
        }
        break;
    }
    case 10:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i];
        }
        break;
    }
    case 11:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i];
        }
        break;
    }
    case 12:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i];
        }
        break;
    }
    case 13:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i];
        }
        break;
    }
    case 14:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i];
        }
        break;
    }
    case 15:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i];
        }
        break;
    }
    case 16:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i] + src15[i];
        }
        break;
    }
    case 17:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i] + src15[i] + src16[i];
        }
        break;
    }
    case 18:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i] + src15[i] + src16[i] + src17[i];
        }
        break;
    }
    case 19:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i] + src15[i] + src16[i] + src17[i] + src18[i];
        }
        break;
    }
    case 20:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i] + src15[i] + src16[i] + src17[i] + src18[i] + src19[i];
        }
        break;
    }
    default:
        std::cerr << "Unknown num_blocks: " << num_blocks << std::endl;
        break;
    }
}

int main(int argc, char **argv)
{
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    //LOG(INFO) << "glog initialized.";
    Send_Ops test(12, 3, {2,3,2});
    test.generate_ops();
    test.print_ops();
    google::ShutdownGoogleLogging();
}