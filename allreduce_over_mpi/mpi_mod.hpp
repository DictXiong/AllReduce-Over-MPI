//start of flextree mod
#ifndef FlexTree_MPI
#define FlexTree_MPI

#if defined(c_plusplus) || defined(__cplusplus)
#include<iostream>

static int FT_enabled()
{
    std::cout << "FlexTree enabled";
    return 0;
}

#include<sstream>
#include<fstream>
#include<vector>
#include<string.h>
#include<thread>
#include<stdlib.h>
#ifndef OMPI_MPI_H
#include<mpi.h>
#include<glog/logging.h>
const int INF = 0x3F3F3F3F;
#endif

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

//#define FT_DEBUG

//#define SHOW_TIME // 显示更多的时间调试信息
#ifdef SHOW_TIME
double _time_base;
#define TIME_RESET() do {_time_base=MPI_Wtime();} while (false)
#define TIME_LOG_IF(exp, note) do {LOG_IF(INFO,exp)<<MPI_Wtime()-_time_base<<" :: "<<note;} while (false)
#endif

// Op
class Operation
{
public:    
    size_t peer;
    std::vector<size_t> blocks;
    /**
     * Operation 类构造函数. 用于 Tree AllReduce.
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
    /**
     * Operation 类构造函数.
     *  
     * @param _peer 接受/发送操作的对象
     * @param _block 接收/发送的数据块的编号
     */
    Operation(size_t _peer, size_t _block): peer(_peer)
    {
        blocks.push_back(_block);
    }
};

// lonely 的意思是: 被树孤立的. 在 ar 过程中, lonely 节点的数据会不按照 stages 来进行, 而是与树的 ar 过程同步并行.
// 为什么不在构造时直接使用 ft_ctx: 因为 ft_ctx 是与 mpi 强耦合的一个东西, 但是 operations 以及拓扑的生成应当只和所需要的这三个参数有关系, 不要和 mpi 扯上关系.
class Operations
{
public:
    std::vector<size_t> stages;
    size_t total_peers, node_label, num_lonely, num_split;
public:
    std::vector<std::vector<Operation>> ops;
    std::vector<Operation> lonely_ops;
    /**
     * Operations 类的构造函数
     * 
     * @param _total_peers 参与计算的总节点数
     * @param _node_label 当前节点的编号
     * @param _stages 一个向量, 记录了 AllReduce 树自下而上每一层的宽度. 注意积 + {@code _num_lonely} 应当等于 {@code _total_peers}.
     * @param _num_lonely 孤立节点的数量
     */ 
    Operations(const size_t &_total_peers, const size_t &_num_lonely, const size_t &_node_label, const std::vector<size_t> &_stages): total_peers(_total_peers), node_label(_node_label), stages(_stages), num_lonely(_num_lonely), num_split(_total_peers - _num_lonely)
    {

        size_t pi = 1;
        for (const auto &i:_stages)
        {
            pi *= i;
        }
    }
    // 生成拓扑, 要求子类实现
    virtual void generate_ops() = 0;
    // 打印拓扑
    virtual void print_ops()const
    {
        std::cout << typeid(*this).name() << " of node " << node_label << " in total " << total_peers << " peers: " << std::endl;
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
        if (num_lonely != 0)
        {
            std::cout << "and " << num_lonely << " lonely node(s):" << std::endl;
            std::cout << "┕ lonely";
            for (const auto &j:lonely_ops)
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
        if (node_label < num_split)
        {
            // 当前组内成员的编号的间距
            size_t gap = 1;
            for (auto i:stages)
            {
                std::vector<Operation> stage_ops;
                // 当前组内编号最小的成员
                size_t left_peer = node_label / (gap * i) * (gap * i) + node_label % gap;
                for (size_t j = 0; j < i; j++)
                {
                    stage_ops.emplace_back(left_peer, num_split, gap * i);
                    left_peer += gap;
                }
                ops.push_back(stage_ops);
                gap *= i;
            }
        }
        else 
        {
            for (size_t i = 0; i < num_split; i++)
            {
                lonely_ops.emplace_back(i, i);
            }
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
        if (node_label < num_split)
        {
            // 当前组内成员的编号的间距
            size_t gap = 1;
            for (auto i:stages)
            {
                std::vector<Operation> stage_ops;
                Operation op_template(node_label, num_split, gap * i);
                // 当前组内编号最小的成员
                size_t left_peer = node_label / (gap * i) * (gap * i) + node_label % gap;
                for (size_t j = 0; j < i; j++)
                {
                    op_template.peer = left_peer;
                    stage_ops.emplace_back(op_template);
                    left_peer += gap;
                }
                ops.push_back(stage_ops);
                gap *= i;
            }
            for (size_t i = num_split; i < total_peers; i++)
            {
                lonely_ops.emplace_back(i, node_label);
            }
        }
    }
};

class FlexTree_Context
{
public:
    size_t num_nodes, node_label, num_lonely, data_size, num_split, split_size, data_size_aligned, type_size;
    bool has_lonely;
    FlexTree_Context(const MPI_Comm &_comm, const MPI_Datatype &_datatype, const size_t &_count, const size_t &_num_lonely = 0)
    {
        int tmp;
        MPI_Comm_size(_comm, &tmp);
        num_nodes = tmp;
        MPI_Comm_rank(_comm, &tmp);
        node_label = tmp;
        num_lonely = _num_lonely;
        data_size = _count;
        num_split = num_nodes - num_lonely;
        split_size = (data_size + num_nodes - 1) / num_nodes; // aligned
        data_size_aligned = split_size * num_nodes;
        MPI_Type_size(_datatype, &tmp);
        type_size = tmp;
        //last_split_size = split_size - (data_size_aligned - data_size);
        // 为什么不能用 last_split_size 呢? 是因为最后一块大小可能为 0, 而且有可能倒数好几块都是 0!!! 为了对齐, 付出的代价可能是好几块. 比如说 10 个节点同步一个大小为 1 的数据块, 当然十块有九块都是空了.
        has_lonely = (num_lonely > 0);
    }
    void show_context() const
    {
        std::cout << "num_nodes=" << num_nodes << ", node_label=" << node_label << ", num_lonely=" << num_lonely << ", data_size=" << data_size << ", num_split=" << num_split << ", split_size=" << split_size << ", data_size_aligned=" << data_size_aligned << ", type_size=" << type_size << ", has_lonely=" << has_lonely << std::endl;
    }
};

const size_t MAX_NUM_BLOCKS = 20;
template<class DataType> 
static void reduce_sum(const DataType **src, DataType *dst, const int &num_blocks, const size_t &num_elements)
{
#ifdef FT_DEBUG
    //std::cout << "reduce_sum called, ele size = " << sizeof(**src) << std::endl;
#endif
    if (num_blocks <= 1) return;
#define PARALLEL_THREAD 14
    const DataType *src0 = src[0];
    const DataType *src1 = src[1];
    const DataType *src2 = src[2];
    const DataType *src3 = src[3];
    const DataType *src4 = src[4];
    const DataType *src5 = src[5];
    const DataType *src6 = src[6];
    const DataType *src7 = src[7];
    const DataType *src8 = src[8];
    const DataType *src9 = src[9];
    const DataType *src10 = src[10];
    const DataType *src11 = src[11];
    const DataType *src12 = src[12];
    const DataType *src13 = src[13];
    const DataType *src14 = src[14];
    const DataType *src15 = src[15];
    const DataType *src16 = src[16];
    const DataType *src17 = src[17];
    const DataType *src18 = src[18];
    const DataType *src19 = src[19];

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

template<class DataType> 
static void reduce_band(const DataType **src, DataType *dst, const int &num_blocks, const size_t &num_elements)
{
#ifdef FT_DEBUG
    //std::cout << "reduce_band called, ele size = " << sizeof(**src) << std::endl;
#endif
    if (num_blocks <= 1) return;
#define PARALLEL_THREAD 14
    const DataType *src0 = src[0];
    const DataType *src1 = src[1];
    const DataType *src2 = src[2];
    const DataType *src3 = src[3];
    const DataType *src4 = src[4];
    const DataType *src5 = src[5];
    const DataType *src6 = src[6];
    const DataType *src7 = src[7];
    const DataType *src8 = src[8];
    const DataType *src9 = src[9];
    const DataType *src10 = src[10];
    const DataType *src11 = src[11];
    const DataType *src12 = src[12];
    const DataType *src13 = src[13];
    const DataType *src14 = src[14];
    const DataType *src15 = src[15];
    const DataType *src16 = src[16];
    const DataType *src17 = src[17];
    const DataType *src18 = src[18];
    const DataType *src19 = src[19];

    switch (num_blocks)
    {
    case 2:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i];
        }
        break;
    }
    case 3:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i];
        }
        break;
    }
    case 4:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i];
        }
        break;
    }
    case 5:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i];
        }
        break;
    }
    case 6:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i];
        }
        break;
    }
    case 7:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i];
        }
        break;
    }
    case 8:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i];
        }
        break;
    }
    case 9:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i];
        }
        break;
    }
    case 10:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i];
        }
        break;
    }
    case 11:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i] & src10[i];
        }
        break;
    }
    case 12:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i] & src10[i] & src11[i];
        }
        break;
    }
    case 13:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i] & src10[i] & src11[i] & src12[i];
        }
        break;
    }
    case 14:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i] & src10[i] & src11[i] & src12[i] & src13[i];
        }
        break;
    }
    case 15:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i] & src10[i] & src11[i] & src12[i] & src13[i] & src14[i];
        }
        break;
    }
    case 16:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i] & src10[i] & src11[i] & src12[i] & src13[i] & src14[i] & src15[i];
        }
        break;
    }
    case 17:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i] & src10[i] & src11[i] & src12[i] & src13[i] & src14[i] & src15[i] & src16[i];
        }
        break;
    }
    case 18:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i] & src10[i] & src11[i] & src12[i] & src13[i] & src14[i] & src15[i] & src16[i] & src17[i];
        }
        break;
    }
    case 19:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i] & src10[i] & src11[i] & src12[i] & src13[i] & src14[i] & src15[i] & src16[i] & src17[i] & src18[i];
        }
        break;
    }
    case 20:
    {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
        for (size_t i = 0; i < num_elements; ++i)
        {
            dst[i] = src0[i] & src1[i] & src2[i] & src3[i] & src4[i] & src5[i] & src6[i] & src7[i] & src8[i] & src9[i] & src10[i] & src11[i] & src12[i] & src13[i] & src14[i] & src15[i] & src16[i] & src17[i] & src18[i] & src19[i];
        }
        break;
    }
    default:
        std::cerr << "Unknown num_blocks: " << num_blocks << std::endl;
        break;
    }
}

// 单纯的发送, 只负责安排工作, 不等待工作完成.
static size_t handle_send(const MPI_Comm &comm, const MPI_Datatype &datatype, const std::vector<Operation> *ops, const void *data, const FlexTree_Context &ft_ctx, MPI_Request request[])
{

    size_t start;
    size_t request_index = 0;

    for (const auto &i : *ops)
    {
        if (LIKELY(ft_ctx.node_label != i.peer))
        {
            for (const auto &j : i.blocks)
            {
                start = ft_ctx.split_size * j;
                //LOG_IF(INFO, node_label == 4) << "##4 send " << j << " which is " << start << "+" << count << " to " << i.peer ;
                
                // 如果当前块的末尾计算值大于实际的总数据块大小 (超出)
                if (UNLIKELY(start + ft_ctx.split_size > ft_ctx.data_size))
                {
                    // 如果当前块的起始位置没有超过实际总数据块大小
                    if (start < ft_ctx.data_size)
                    {
                        MPI_Isend(data + start * ft_ctx.type_size, ft_ctx.data_size - start, datatype, i.peer, 0, comm, &request[request_index++]); // 此处的tag暂时先打0
#ifdef FT_DEBUG
                        std::cout << ft_ctx.node_label << " send " << j << " which is " << start << "+" << ft_ctx.data_size - start << " to " << i.peer << ", element size = " << ft_ctx.type_size << std::endl;
#endif
                    }
                    // 否则根本不发送 (因为块为空)
                    else
                    {
#ifdef FT_DEBUG
                        std::cout << ft_ctx.node_label << " will not send " << j << " which starts from " << start << " to " << i.peer << " because it's empty." << std::endl;
#endif
                    }
                }
                else 
                {
#ifdef FT_DEBUG
                    std::cout << ft_ctx.node_label << " send " << j << " which is " << start << "+" << ft_ctx.split_size << " to " << i.peer << ", element size = " << ft_ctx.type_size << std::endl;
#endif
                    MPI_Isend(data + start * ft_ctx.type_size, ft_ctx.split_size, datatype, i.peer, 0, comm, &request[request_index++]); // 此处的tag暂时先打0
                }
            }
        }
    }
    return request_index;
}

// 同上, 只负责安排工作, 不等待工作完成.
// accordingly 参数的含义是, 如果为 true, 那么把数据块写到 buffer 中对应的位置去; 如果为 false, 那么直接平铺在 buffer 中.
static size_t handle_recv(const MPI_Comm &comm, const MPI_Datatype &datatype, const std::vector<Operation> *ops, void *buffer, const FlexTree_Context &ft_ctx, const bool &accordingly, MPI_Request request[])
{

    size_t start = 0;
    size_t request_index = 0;

    for (const auto &i : *ops)
    {
        if (LIKELY(ft_ctx.node_label != i.peer))
        {
            for (const auto &j : i.blocks)
            {
                size_t split_accord_start = ft_ctx.split_size * j;
                if (accordingly) 
                {
                    start = split_accord_start;
                }

                // 如果当前块的末尾计算值大于实际的总数据块大小 (超出)
                if (UNLIKELY(split_accord_start + ft_ctx.split_size > ft_ctx.data_size))
                {
                    // 如果当前块的起始位置没有超过实际总数据块大小
                    if (split_accord_start < ft_ctx.data_size)
                    {
#ifdef FT_DEBUG
                        std::cout << ft_ctx.node_label << " recv " << j << " which will be placed to " << start << "+" << ft_ctx.data_size - split_accord_start << " from " << i.peer << ", element size = " << ft_ctx.type_size << std::endl;
#endif
                        MPI_Irecv(buffer + start * ft_ctx.type_size, ft_ctx.data_size - split_accord_start, datatype, i.peer, 0, comm, &request[request_index++]); // 此处的tag暂时先打0
                    }
                    // 否则根本不接收 (因为块为空)
                    else
                    {
#ifdef FT_DEBUG
                        std::cout << ft_ctx.node_label << " will not recv " << j << " from " << i.peer << " because it's empty" << std::endl;
#endif
                    }
                }
                else
                {
#ifdef FT_DEBUG
                    std::cout << ft_ctx.node_label << " recv " << j << " which will be placed to " << start << "+" << ft_ctx.split_size << " from " << i.peer << ", element size = " << ft_ctx.type_size << std::endl;
#endif
                    MPI_Irecv(buffer + start * ft_ctx.type_size, ft_ctx.split_size, datatype, i.peer, 0, comm, &request[request_index++]); // 此处的tag暂时先打0
                }
                
                if (!accordingly)
                {
                    start += ft_ctx.split_size;
                }
            }
        }
    }
    return request_index;
}

// 负责进行加和, 然后放到指定的位置上去. 注意会自动包含自己的那块data.
// 这里的 dest 是一块和 data 大小/结构相同的一块内存. 进行 reduce 的时候, 会把结果对应地放进 dest 去. 注意 dest 不可以是 null.
static void handle_reduce(const MPI_Datatype &datatype, const MPI_Op &op, const std::vector<size_t> *blocks, void *buffer, const void *data, void *dest, const FlexTree_Context &ft_ctx, const size_t &num_peers, void *extra_buffer = nullptr, const size_t &extra_peers = 0)
{
    if (dest == nullptr)
    {
        std::cerr << "I can't reduce to null. Aborted." << std::endl;
        exit(1);
    }
    const size_t peer_gap = blocks->size() * ft_ctx.split_size;
    const void **src = (const void**)(new char*[num_peers + extra_peers + 10]);
    void *dst;
    for (int i = 0; i < num_peers + 2; i++)
    {
        src[i] = nullptr;
    }
    for (auto i = blocks->begin(); i != blocks->end(); i++)
    {
        size_t start = ft_ctx.split_size * (*i);
        size_t src_index = 1;
        src[0] = data + start * ft_ctx.type_size;
        dst = dest + start * ft_ctx.type_size;
        size_t split_size = ft_ctx.split_size;
        // 如果当前块的理论结尾位置超过了实际的块大小
        if (UNLIKELY(start + ft_ctx.split_size > ft_ctx.data_size))
        {
            // 如果当前块的起始位置没有超过实际总数据块大小
            if (start < ft_ctx.data_size)
            {
                split_size = ft_ctx.data_size - start;
            }
            else
            {
#ifdef FT_DEBUG
                std::cout << ft_ctx.node_label << " will not reduce " << *i << " because it's empty." << std::endl;
#endif
                continue; // 当前块实际大小为零, 直接溜了.
            }
        }
#ifdef FT_DEBUG
        std::cout << ft_ctx.node_label << " reduce " << *i << " which size is " << split_size << ", element size = " << ft_ctx.type_size << std::endl;
#endif
        start = (i - blocks->begin()) * ft_ctx.split_size;
        for (size_t j = 0; j < num_peers; j++)
        {
            src[src_index++] = buffer + start * ft_ctx.type_size;
#ifdef FT_DEBUG
            std::cout << "  --" << ft_ctx.node_label << " will reduce data at " << start << std::endl;
#endif
            start += peer_gap;
        }
        start = (i - blocks->begin()) * ft_ctx.split_size;
        for (size_t j = 0; j < extra_peers; j++)
        {
            src[src_index++] = extra_buffer + start * ft_ctx.type_size;
            start += peer_gap;
        }
        
        if (op == MPI_SUM)
        {
            if (datatype == MPI_UINT8_T) reduce_sum((const uint8_t**)src, (uint8_t*)dst, src_index, split_size);
            else if (datatype == MPI_INT8_T) reduce_sum((const int8_t**)src, (int8_t*)dst, src_index, split_size);
            else if (datatype == MPI_UINT16_T) reduce_sum((const uint16_t**)src, (uint16_t*)dst, src_index, split_size);
            else if (datatype == MPI_INT16_T) reduce_sum((const int16_t**)src, (int16_t*)dst, src_index, split_size);
            else if (datatype == MPI_INT32_T) reduce_sum((const int32_t**)src, (int32_t*)dst, src_index, split_size);
            else if (datatype == MPI_INT64_T) reduce_sum((const int64_t**)src, (int64_t*)dst, src_index, split_size);
            else if (datatype == MPI_FLOAT) reduce_sum((const float**)src, (float*)dst, src_index, split_size);
            else if (datatype == MPI_DOUBLE) reduce_sum((const double**)src, (double*)dst, src_index, split_size);
            else if (datatype == MPI_C_BOOL) reduce_sum((const bool**)src, (bool*)dst, src_index, split_size);
            else if (datatype == MPI_LONG_LONG_INT) reduce_sum((const long long int**)src, (long long int*)dst, src_index, split_size);
            else if (datatype == MPI_LONG_LONG) reduce_sum((const long long**)src, (long long*)dst, src_index, split_size);
            else 
            {
                char name[20];
                int name_len;
                MPI_Type_get_name(datatype, name, &name_len);
                name[name_len] = '\0';
                std::string s = name;
                std::cerr << "Type " << s << " is not supported in MPI mode." << std::endl;
                exit(1);
            }
        }
        else if (op == MPI_BAND)
        {
            if (datatype == MPI_UINT8_T) reduce_band((const uint8_t**)src, (uint8_t*)dst, src_index, split_size);
            else if (datatype == MPI_INT8_T) reduce_band((const int8_t**)src, (int8_t*)dst, src_index, split_size);
            else if (datatype == MPI_UINT16_T) reduce_band((const uint16_t**)src, (uint16_t*)dst, src_index, split_size);
            else if (datatype == MPI_INT16_T) reduce_band((const int16_t**)src, (int16_t*)dst, src_index, split_size);
            else if (datatype == MPI_INT32_T) reduce_band((const int32_t**)src, (int32_t*)dst, src_index, split_size);
            else if (datatype == MPI_INT64_T) reduce_band((const int64_t**)src, (int64_t*)dst, src_index, split_size);
            else if (datatype == MPI_LONG_LONG_INT) reduce_band((const long long int**)src, (long long int*)dst, src_index, split_size);
            else if (datatype == MPI_LONG_LONG) reduce_band((const long long**)src, (long long*)dst, src_index, split_size);
            else 
            {
                char name[20];
                int name_len;
                MPI_Type_get_name(datatype, name, &name_len);
                name[name_len] = '\0';
                std::string s = name;
                std::cerr << "Type " << s << " is not supported in MPI mode." << std::endl;
                exit(1);
            }
        }
        else 
        {
            std::cerr << "Unsupported op " << op << std::endl;
            exit(1);
        }
    }
    delete[] src;
    src = nullptr;
}

// 从环境变量获取每一层宽度
static std::vector<size_t> get_stages(const size_t &num_nodes)
{
    std::string FT_TOPO; 
    auto FT_TOPO_raw = getenv("FT_TOPO");
    std::vector<size_t> ans;
    size_t pi = 1;
    int tmp;
    if (FT_TOPO_raw != nullptr)
    {
        FT_TOPO = FT_TOPO_raw;
    }
    if (FT_TOPO.empty())
    {
        ans = {num_nodes};
    }
    else 
    {
        for (char &i : FT_TOPO)
        {
            if (i == ',') i = ' ';
        }
        std::istringstream ss(FT_TOPO);
        while(!ss.eof())
        {
            ss >> tmp;
            ans.push_back(tmp);
            pi *= tmp;
        }
        if (pi != num_nodes)
        {
            std::cerr << "invalid FT_TOPO " << FT_TOPO << std::endl;
            exit(1);
        }
    }
#ifdef FT_DEBUG
    std::cout << "FlexTree topo is ";
    for (auto i:ans)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
#endif
    return ans;
}

static bool comm_only = false;
static void *recv_buffer = nullptr; //必须初始化

// 如果需要原地 ar, 那么将 data 置为 nullptr.
static void tree_allreduce(const MPI_Datatype &datatype, const MPI_Op &op, const MPI_Comm &comm, const void *data, void *dst, const FlexTree_Context &ft_ctx, const std::vector<size_t> &stages)
{
#ifdef FT_DEBUG
    //std::cout << "FT DEBUG: inside treeallre: op " << op << "; len = " << len << "; total = " << num_nodes << "; datatype = " << datatype << std::endl;
    std::cout << "LOOK HERE (TMP): " << stages.size();
#endif
    if (data == nullptr)
    {
        data = dst;
    }
    //LOG_IF(WARNING, node_label == 0) << "gathering start";
    Send_Ops send_ops(ft_ctx.num_nodes, ft_ctx.num_lonely, ft_ctx.node_label, stages);
    Recv_Ops recv_ops(ft_ctx.num_nodes, ft_ctx.num_lonely, ft_ctx.node_label, stages);
    send_ops.generate_ops();
    recv_ops.generate_ops();
    MPI_Comm sub_comm = comm;
    const size_t MAX_COMM_SIZE = 2 * (ft_ctx.num_split - 1) * (ft_ctx.num_split);
    size_t request_index = 0;
    MPI_Request *requests = new MPI_Request[MAX_COMM_SIZE];
    MPI_Status *status = new MPI_Status[MAX_COMM_SIZE];
    size_t lonely_request_index = 0;
    MPI_Request *lonely_requests;
    int tmp;
#ifdef SHOW_TIME
    TIME_RESET();
#endif
    if (ft_ctx.node_label < ft_ctx.num_split)
    {
        if (ft_ctx.has_lonely)
        {
            // 如果要用, 则必须修改.
            //lonely_requests = new MPI_Request[ft_ctx.num_lonely << 1];
            //MPI_Comm_split(comm, 0, ft_ctx.node_label, &sub_comm); // 这个 0 是 magic number, 用来标注本组的颜色.
            // lonely_request_index = handle_recv(comm, datatype, &(recv_ops.lonely_ops), data + len * type_size, ft_ctx, false, lonely_requests);
        }
        for (size_t i = 0; i != stages.size(); i++)
        {
            // 这一步判断是为什么呢? 是因为, 函数不会试图修改data的内容, 已经reduce的数据将会放在dst中; 而除了第一步之外, 发送的都是reduce后的数据, 所以第一步需要单独提出来.
            if (i == 0)
            {
                request_index = handle_send(comm, datatype, &(send_ops.ops[i]), data, ft_ctx, requests + request_index); //这里顺便重置了 index
            }
            else
            {
                request_index = handle_send(comm, datatype, &(send_ops.ops[i]), dst, ft_ctx, requests + request_index); //这里顺便重置了 index
            }
            tmp = handle_recv(comm, datatype, &(recv_ops.ops[i]), recv_buffer, ft_ctx, false, requests + request_index);
#ifdef FT_DEBUG
            //std::cout << "FT DEBUG: start to send/recv" << std::endl;
#endif
            MPI_Waitall(tmp, requests + request_index, status);
#ifdef FT_DEBUG
            //std::cout << "FT DEBUG: complete send/recv" << std::endl;
#endif
            if (lonely_request_index == 0 || i != stages.size() - 1)
            {
                // 这里判断的原因和上面一样
                if (i == 0)
                {
                    handle_reduce(datatype, op, &(recv_ops.ops[i][0].blocks), recv_buffer, data, dst, ft_ctx, recv_ops.ops[i].size() - 1);
                }
                else
                {
                    handle_reduce(datatype, op, &(recv_ops.ops[i][0].blocks), recv_buffer, dst, dst, ft_ctx, recv_ops.ops[i].size() - 1);
                }
            }
            else
            {
                MPI_Waitall(lonely_request_index, lonely_requests, status);
#ifdef SHOW_TIME
                TIME_LOG_IF(node_label == 0, "node 0 lonely gather finished");
#endif SHOW_TIME
                // 如果要用, 则必须修改. handle_reduce(datatype, op, &(recv_ops.ops[i][0].blocks), recv_buffer, data, dst, ft_ctx, recv_ops.ops[i].size() - 1, data + ft_ctx.len * ft_ctx.type_size, ft_ctx.num_lonely);
            }
            MPI_Waitall(request_index, requests, status);
            MPI_Barrier(sub_comm);
        }
#ifdef SHOW_TIME
            TIME_LOG_IF(node_label == 0, "(left) FT gather finished");
#endif SHOW_TIME
        if (ft_ctx.has_lonely) MPI_Barrier(comm);
        //LOG_IF(WARNING, node_label == 0) << "gathering done";
#ifdef SHOW_TIME
        TIME_RESET();
#endif
#ifdef FT_DEBUG
        std::cout << "-------- FT DEBUG: complete reduce --------" << std::endl;
#endif
        if (ft_ctx.has_lonely)
        {
            // 测试是否和内存锁有关系
            //size_t start = len / num_split * node_label;
            //memcpy(recv_buffer + start, data + start, sizeof(DataType) * len / num_split);
            //lonely_request_index = handle_send(&(recv_ops.lonely_ops), recv_buffer, len, num_split, node_label, lonely_requests);
            // end
            //lonely_request_index = handle_send(&(recv_ops.lonely_ops), data, len, num_split, node_label, lonely_requests);
        }
        for (int i = stages.size() - 1; i >= 0; i--)
        {
            if (i == 0 && ft_ctx.has_lonely)
            {
                // 如果要用, 则必须修改. lonely_request_index = handle_send(comm, datatype, &(recv_ops.lonely_ops), data, ft_ctx, lonely_requests);
            }
            request_index = handle_send(comm, datatype, &(recv_ops.ops[i]), dst, ft_ctx, requests);
            request_index += handle_recv(comm, datatype, &(send_ops.ops[i]), dst, ft_ctx, true, requests + request_index);
            MPI_Waitall(request_index, requests, status);
            MPI_Barrier(sub_comm);
        }
#ifdef SHOW_TIME
                TIME_LOG_IF(node_label == 0, "(left comm) FT broadcast finished");
#endif SHOW_TIME
        if (ft_ctx.has_lonely)
        {
            MPI_Waitall(lonely_request_index, lonely_requests, status);
#ifdef SHOW_TIME
            TIME_LOG_IF(node_label == 0, "node 0 lonely broadcast finished");
#endif SHOW_TIME
            MPI_Barrier(comm);
            delete[] lonely_requests;
            lonely_requests = nullptr;
        }
    }
    else 
    {
        lonely_requests = new MPI_Request[ft_ctx.num_split << 2];
        MPI_Comm_split(comm, 1, ft_ctx.node_label, &sub_comm); // 这个 1 是 magic number, 用来标注本组的颜色.
        //LOG(WARNING) << "LONELY send start";
        // 如果要用, 则必须修改. lonely_request_index = handle_send(comm, datatype, &(send_ops.lonely_ops), data, ft_ctx, lonely_requests);
        MPI_Waitall(lonely_request_index, lonely_requests, status);
#ifdef SHOW_TIME
        TIME_LOG_IF(true, "(right) lonely send finished");
#endif SHOW_TIME
        //LOG(WARNING) << "LONELY send done";
        MPI_Barrier(comm);
#ifdef SHOW_TIME
        TIME_RESET();
#endif
        //LOG(WARNING) << "LONELY recv start";
        // 如果要用, 则必须修改. lonely_request_index = handle_recv(comm, datatype, &(send_ops.lonely_ops), data, ft_ctx, true, lonely_requests);
        MPI_Waitall(lonely_request_index, lonely_requests, status);
#ifdef SHOW_TIME
        TIME_LOG_IF(true, "lonely recv finished");
#endif SHOW_TIME
        MPI_Barrier(comm);
        delete[] lonely_requests;
        lonely_requests = nullptr;
    }
    delete[] requests;
    delete[] status;
    requests = nullptr;
    status = nullptr;
#ifdef FT_DEBUG
    std::cout << "-------- FT DEBUG: complete allreduce --------" << std::endl;
#endif
    //LOG_IF(WARNING, node_label == 0) << "broadcast done";
#ifdef FT_DEBUG
    std::cout << "WHY HERE: " << ((int32_t*)dst)[9] << " " << ((int32_t*)dst)[12] << std::endl;
#endif
}

// NOTE: 可别把这个buffer给私自delete了
static void* flextree_register_the_buffer(size_t _size)
{
    static void* buffer;
    static size_t size = 0;
    if (_size > size)
    {
        if (size != 0)
        {
            delete[] buffer;
            buffer = nullptr;
        }
#ifdef FT_DEBUG
        std::cout << "registered a buffer of " << _size << std::endl;
#endif
        buffer = (void*)(new char[_size]);
        size = _size;
    }
    return buffer;
}

#ifndef OMPI_MPI_H
int MPI_Allreduce_FT(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
#else
static int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
#endif
{
#ifdef FT_DEBUG
    std::cout << "FlexTree AR called" << std::endl;
#endif
    const FlexTree_Context ft_ctx(comm, datatype, count);
#ifdef FT_DEBUG
    if (ft_ctx.node_label == ft_ctx.num_nodes - 2) ft_ctx.show_context();
#endif

    if (ft_ctx.num_nodes <= 1)
    {
        if (sendbuf != MPI_IN_PLACE)
        {
            memcpy(recvbuf, sendbuf, count * ft_ctx.type_size);
        }
        return 0;
    }

    recv_buffer = flextree_register_the_buffer(ft_ctx.data_size_aligned * ft_ctx.type_size);
    auto stages = get_stages(ft_ctx.num_nodes);
    
    // MPI_IN_PLACE
    if (sendbuf == MPI_IN_PLACE)
    {
        tree_allreduce(datatype, op, comm, nullptr, recvbuf, ft_ctx, stages);
    }
    else 
    {
        tree_allreduce(datatype, op, comm, sendbuf, recvbuf, ft_ctx, stages);
    }
#ifdef FT_DEBUG
    std::cout << "FlexTree AR finished" << std::endl;
#endif
    return 0;
}

#endif //end if of check c++
#endif
//end of flextree mod