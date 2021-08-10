#include<iostream>
#include<sstream>
#include<fstream>
#include<vector>
#include<string.h>
#include<thread>
#include<mpi.h>
#include "glog/logging.h"

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

typedef float DataType;
const int INF = 0x3F3F3F3F;

#define SHOW_TIME // 显示更多的时间调试信息
#ifdef SHOW_TIME
double _time_base;
#define TIME_RESET() do {_time_base=MPI_Wtime();} while (false)
#define TIME_LOG_IF(exp, note) do {LOG_IF(INFO,exp)<<MPI_Wtime()-_time_base<<" :: "<<note;} while (false)
#endif

// util
template<typename T>
void write_vector_to_file(std::vector<T> vec, std::string filename)
{
    std::ofstream f(filename, std::ios::out);
    for (auto &i : vec)
    {
        f << i << std::endl;
    }
    f.close();
}

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
    Operations(size_t _total_peers, size_t _num_lonely, size_t _node_label, std::vector<size_t> _stages): total_peers(_total_peers), node_label(_node_label), stages(_stages), num_lonely(_num_lonely), num_split(_total_peers - _num_lonely)
    {
        CHECK_GT(_total_peers, 0);
        CHECK_GE(_node_label, 0);
        CHECK_LT(_node_label, _total_peers);
        CHECK_GE(_num_lonely, 0);
        CHECK_GT(num_split, 0);
        size_t pi = 1;
        for (const auto &i:_stages)
        {
            CHECK_GT(i, 0);
            pi *= i;
        }
        CHECK_EQ(pi + _num_lonely, _total_peers);
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

const size_t MAX_NUM_BLOCKS = 20;
void reduce_sum(DataType **src, int num_blocks, size_t num_elements)
{
    CHECK_LE(num_blocks, MAX_NUM_BLOCKS);
#define PARALLEL_THREAD 14
    DataType *dst = src[0];
    DataType *src0 = src[0];
    DataType *src1 = src[1];
    DataType *src2 = src[2];
    DataType *src3 = src[3];
    DataType *src4 = src[4];
    DataType *src5 = src[5];
    DataType *src6 = src[6];
    DataType *src7 = src[7];
    DataType *src8 = src[8];
    DataType *src9 = src[9];
    DataType *src10 = src[10];
    DataType *src11 = src[11];
    DataType *src12 = src[12];
    DataType *src13 = src[13];
    DataType *src14 = src[14];
    DataType *src15 = src[15];
    DataType *src16 = src[16];
    DataType *src17 = src[17];
    DataType *src18 = src[18];
    DataType *src19 = src[19];

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
        LOG(FATAL) << "Unknown num_blocks: " << num_blocks;
        break;
    }
}

// 单纯的发送, 只负责安排工作, 不等待工作完成.
size_t handle_send(std::vector<Operation> *ops, DataType *data, size_t len, size_t num_split, size_t node_label, MPI_Request request[])
{
    CHECK_NOTNULL(ops);
    CHECK_NOTNULL(data);
    CHECK_NOTNULL(request);

    size_t start;
    size_t request_index = 0;

    int count = len / num_split;

    for (const auto &i : *ops)
    {
        if (LIKELY(node_label != i.peer))
        {
            for (const auto &j : i.blocks)
            {
                start = len / num_split * j;
                //LOG_IF(INFO, node_label == 4) << "##4 send " << j << " which is " << start << "+" << count << " to " << i.peer ;
                MPI_Isend(data + start, count, MPI_FLOAT, i.peer, 0, MPI_COMM_WORLD, &request[request_index++]); // 此处的tag暂时先打0
            }
        }
    }
    return request_index;
}

// 同上, 只负责安排工作, 不等待工作完成.
// accordingly 参数的含义是, 如果为 true, 那么把数据块写到 buffer 中对应的位置去; 如果为 false, 那么直接平铺在 buffer 中.
size_t handle_recv(std::vector<Operation> *ops, DataType *buffer, size_t len, size_t num_split, size_t node_label, bool accordingly, MPI_Request request[])
{
    CHECK_NOTNULL(ops);
    CHECK_NOTNULL(buffer);

    size_t start = 0;
    size_t request_index = 0;

    int count = len / num_split; // 单位数据块的大小

    for (const auto &i : *ops)
    {
        if (LIKELY(node_label != i.peer))
        {
            for (const auto &j : i.blocks)
            {
                if (accordingly) 
                {
                    start = len / num_split * j;
                }
                MPI_Irecv(buffer + start, count, MPI_FLOAT, i.peer, 0, MPI_COMM_WORLD, &request[request_index++]); // 此处的tag暂时先打0
                if (!accordingly)
                {
                    start += len / num_split;
                }
            }
        }
    }
    return request_index;
}

// 负责进行加和, 然后放到指定的位置上去. 注意会自动包含自己的那块data.
void handle_reduce(std::vector<size_t> *blocks, DataType *buffer, DataType *data, size_t len, size_t num_split, size_t num_peers, DataType *extra_buffer = nullptr, size_t extra_peers = 0)
{
    const int block_size = len / num_split;
    const size_t peer_gap = blocks->size() * block_size;
    DataType **src = new DataType*[num_peers + 2];
    for (int i = 0; i < num_peers + 2; i++)
    {
        src[i] = nullptr;
    }
    for (auto i = blocks->begin(); i != blocks->end(); i++)
    {
        size_t start = len / num_split * (*i);
        size_t src_index = 1;
        src[0] = data + start;
        start = (i - blocks->begin()) * block_size;
        for (size_t j = 0; j < num_peers; j++)
        {
            src[src_index++] = buffer + start;
            start += peer_gap;
        }
        start = (i - blocks->begin()) * block_size;
        for (size_t j = 0; j < extra_peers; j++)
        {
            src[src_index++] = extra_buffer + start;
            start += peer_gap;
        }
        reduce_sum(src, src_index, block_size);
    }
    delete[] src;
}

bool comm_only = false;
DataType *recv_buffer = nullptr; //必须初始化
// 接收后, 还负责加和
void handle_recv_gather(std::vector<Operation> *ops, DataType *data, size_t len, size_t num_split, size_t node_label)
{
    CHECK_NOTNULL(ops);
    CHECK_NOTNULL(data);

    //DataType *recv_buffer = new DataType[len]; // 创建 recv_buffer
    LOG_IF(FATAL, recv_buffer == nullptr);
    size_t buffer_index = 0;
    const size_t peer_gap = (*ops)[0].blocks.size(); // recv_buffer 中来自同一节点的数据块连续存放, 这是不同节点的相同数据块之间的间距
    const size_t count_peers = (ops->size() == 1 ? 1 : ops->size() - 1); // 要与多少人进行通信
    const size_t comm_size = count_peers * peer_gap; // 总共要进行的通信的次数

    // 用于 MPI_Waitall()
    MPI_Request *request = new MPI_Request[comm_size];
    size_t request_index = 0;
    MPI_Status *status = new MPI_Status[comm_size];

    int count = len / num_split; // 单位数据块的大小
    for (const auto &i : *ops)
    {
        if (LIKELY(node_label != i.peer))
        {
            for (const auto &j : i.blocks)
            {
                MPI_Irecv(recv_buffer + buffer_index, count, MPI_FLOAT, i.peer, 0, MPI_COMM_WORLD, &request[request_index++]); // 此处的tag暂时先打0
                buffer_index += count;
            }
        } 
    }
    MPI_Waitall(request_index, request, status);
    delete[] request;
    delete[] status;

    if (comm_only) return;
    // 通信结束, 开始加和
    DataType **src = new DataType*[MAX_NUM_BLOCKS];
    for (int i = 0; i != MAX_NUM_BLOCKS; i++)
    {
        src[i] = nullptr;
    }
    for (auto i = (*ops)[0].blocks.begin(); i != (*ops)[0].blocks.end(); i++)
    {
        size_t start = len / num_split * (*i);
        src[0] = data + start;
        start = (i - (*ops)[0].blocks.begin()) * count;
        for (size_t j = 0; j != count_peers; j++)
        {
            src[j + 1] = recv_buffer + start;
            start += count * peer_gap;
        }
        reduce_sum(src, count_peers + 1, count);
    }
    delete[] src;
}

// 接收后, 还负责直接覆写数据. 
// accordingly 参数的含义是, 如果为 true, 那么把数据块写到 data 中对应的位置去; 如果为 false, 那么直接平铺在 data 中.
void handle_recv_overwrite(std::vector<Operation> *ops, DataType *data, size_t len, size_t num_split, size_t node_label, bool accordingly)
{
    CHECK_NOTNULL(ops);
    CHECK_NOTNULL(data);

    size_t start = 0;

    const size_t count_peers = ops->size(); // 要与多少人进行通信. 这个值可能大了 1, 因为 ops 里可能包含与自己的通讯, 但是大了好.
    const size_t peer_gap = (*ops)[0].blocks.size(); // 与每人通信的次数
    const size_t comm_size = count_peers * peer_gap; // 总共要进行的通信的次数

    // 用于 MPI_Waitall()
    MPI_Request *request = new MPI_Request[comm_size];
    size_t request_index = 0;
    MPI_Status *status = new MPI_Status[comm_size];

    int count = len / num_split; // 单位数据块的大小

    for (const auto &i : *ops)
    {
        if (LIKELY(node_label != i.peer))
        {
            for (const auto &j : i.blocks)
            {
                if (accordingly) 
                {
                    start = len / num_split * j;
                }
                MPI_Irecv(data + start, count, MPI_FLOAT, i.peer, 0, MPI_COMM_WORLD, &request[request_index++]); // 此处的tag暂时先打0
                if (!accordingly)
                {
                    start += len / num_split;
                }
            }
        }
    }

    MPI_Waitall(request_index, request, status);
    delete[] request;
    delete[] status;
#ifdef SHOW_TIME
    TIME_LOG_IF(node_label == 0 && accordingly == false, "Node 0 lonely recv done");
#endif
    
}

void tree_allreduce(DataType *data, size_t len, size_t num_nodes, size_t num_lonely, size_t node_label, std::vector<size_t> stages)
{
    CHECK_NOTNULL(data);
    CHECK_EQ(0, len % (num_nodes - num_lonely)) << "data length should be an integral multiple of the total bumber of nodes";

    size_t num_split = num_nodes - num_lonely;
    //LOG_IF(WARNING, node_label == 0) << "gathering start";
    Send_Ops send_ops(num_nodes, num_lonely, node_label, stages);
    Recv_Ops recv_ops(num_nodes, num_lonely, node_label, stages);
    send_ops.generate_ops();
    recv_ops.generate_ops();
    MPI_Comm sub_comm = MPI_COMM_WORLD;
    const size_t MAX_COMM_SIZE = 2 * (num_split - 1) * (num_split);
    size_t request_index = 0;
    MPI_Request *requests = new MPI_Request[MAX_COMM_SIZE];
    MPI_Status *status = new MPI_Status[MAX_COMM_SIZE];
    size_t lonely_request_index = 0;
    MPI_Request *lonely_requests;
    int tmp;
#ifdef SHOW_TIME
    TIME_RESET();
#endif
    if (node_label < num_split)
    {
        if (num_lonely > 0)
        {
            lonely_requests = new MPI_Request[num_lonely << 1];
            MPI_Comm_split(MPI_COMM_WORLD, 0, node_label, &sub_comm); // 这个 0 是 magic number, 用来标注本组的颜色.
            lonely_request_index = handle_recv(&(recv_ops.lonely_ops), data + len, len, num_split, node_label, false, lonely_requests);
        }
        for (size_t i = 0; i != stages.size(); i++)
        {
            request_index = handle_send(&(send_ops.ops[i]), data, len, num_split, node_label, requests + request_index); //这里顺便重置了 index
            tmp = handle_recv(&(recv_ops.ops[i]), recv_buffer, len, num_split, node_label, false, requests + request_index);
            MPI_Waitall(tmp, requests + request_index, status);
            if (lonely_request_index == 0 || i != stages.size() - 1)
            {
                handle_reduce(&(recv_ops.ops[i][0].blocks), recv_buffer, data, len, num_split, recv_ops.ops[i].size() - 1);
            }
            else
            {
                MPI_Waitall(lonely_request_index, lonely_requests, status);
#ifdef SHOW_TIME
                TIME_LOG_IF(node_label == 0, "node 0 lonely gather finished");
#endif SHOW_TIME
                handle_reduce(&(recv_ops.ops[i][0].blocks), recv_buffer, data, len, num_split, recv_ops.ops[i].size() - 1, data + len, num_lonely);
            }
            MPI_Waitall(request_index, requests, status);
            MPI_Barrier(sub_comm);
        }
#ifdef SHOW_TIME
            TIME_LOG_IF(node_label == 0, "(left) FT gather finished");
#endif SHOW_TIME
        if (num_lonely > 0) MPI_Barrier(MPI_COMM_WORLD);
        //LOG_IF(WARNING, node_label == 0) << "gathering done";
#ifdef SHOW_TIME
        TIME_RESET();
#endif
        if (num_lonely > 0)
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
            if (i == 0 && num_lonely > 0)
            {
                lonely_request_index = handle_send(&(recv_ops.lonely_ops), data, len, num_split, node_label, lonely_requests);
            }
            request_index = handle_send(&(recv_ops.ops[i]), data, len, num_split, node_label, requests);
            request_index += handle_recv(&(send_ops.ops[i]), data, len, num_split, node_label, true, requests + request_index);
            MPI_Waitall(request_index, requests, status);
            MPI_Barrier(sub_comm);
        }
#ifdef SHOW_TIME
                TIME_LOG_IF(node_label == 0, "(left comm) FT broadcast finished");
#endif SHOW_TIME
        if (num_lonely > 0)
        {
            MPI_Waitall(lonely_request_index, lonely_requests, status);
#ifdef SHOW_TIME
            TIME_LOG_IF(node_label == 0, "node 0 lonely broadcast finished");
#endif SHOW_TIME
            MPI_Barrier(MPI_COMM_WORLD);
            delete[] lonely_requests;
        }
    }
    else 
    {
        lonely_requests = new MPI_Request[num_split << 2];
        MPI_Comm_split(MPI_COMM_WORLD, 1, node_label, &sub_comm); // 这个 1 是 magic number, 用来标注本组的颜色.
        //LOG(WARNING) << "LONELY send start";
        lonely_request_index = handle_send(&(send_ops.lonely_ops), data, len, num_split, node_label, lonely_requests);
        MPI_Waitall(lonely_request_index, lonely_requests, status);
#ifdef SHOW_TIME
        TIME_LOG_IF(true, "(right) lonely send finished");
#endif SHOW_TIME
        //LOG(WARNING) << "LONELY send done";
        MPI_Barrier(MPI_COMM_WORLD);
#ifdef SHOW_TIME
        TIME_RESET();
#endif
        //LOG(WARNING) << "LONELY recv start";
        lonely_request_index = handle_recv(&(send_ops.lonely_ops), data, len, num_split, node_label, true, lonely_requests);
        MPI_Waitall(lonely_request_index, lonely_requests, status);
#ifdef SHOW_TIME
        TIME_LOG_IF(true, "lonely recv finished");
#endif SHOW_TIME
        MPI_Barrier(MPI_COMM_WORLD);
        delete[] lonely_requests;
    }
    delete[] requests;
    delete[] status;
    //LOG_IF(WARNING, node_label == 0) << "broadcast done";
}

void ring_allreduce(DataType *data, size_t len, size_t num_nodes, size_t node_label)
{
    CHECK_NOTNULL(data);
    CHECK_EQ(0, len % num_nodes) << "data length should be an integral multiple of the total bumber of nodes";
    const size_t left = (node_label == 0 ? num_nodes - 1 : node_label - 1);
    const size_t right = (node_label == num_nodes - 1 ? 0 : node_label + 1);
    size_t block_send = node_label;
    size_t block_recv = left;
    const size_t MAX_COMM_SIZE = 4;
    size_t request_index = 0;
    MPI_Request *requests = new MPI_Request[MAX_COMM_SIZE];
    MPI_Status *status = new MPI_Status[MAX_COMM_SIZE];
    
    //LOG_IF(WARNING, node_label == 0) << "gathering start";
    for (size_t i = 0; i != num_nodes - 1; i++)
    {
        std::vector<Operation> send_ops = {Operation(right, block_send)};
        std::vector<Operation> recv_ops = {Operation(left, block_recv)};
        request_index = handle_send(&send_ops, data, len, num_nodes, node_label, requests);
        std::thread recv_thread(handle_recv_gather, &recv_ops, data, len, num_nodes, node_label);
        MPI_Waitall(request_index, requests, status); // 这个地方暂时采用这种新旧混合的写法. 之后有空了再改吧.
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD);
        block_send = (block_send == 0 ? num_nodes - 1 : block_send - 1);
        block_recv = (block_recv == 0 ? num_nodes - 1 : block_recv - 1);
    }
    //LOG_IF(WARNING, node_label == 0) << "gathering done";
    for (size_t i = 0; i != num_nodes - 1; i++)
    {
        std::vector<Operation> send_ops = {Operation(right, block_send)};
        std::vector<Operation> recv_ops = {Operation(left, block_recv)};
        request_index = handle_send(&send_ops, data, len, num_nodes, node_label, requests);
        std::thread recv_thread(handle_recv_overwrite, &recv_ops, data, len, num_nodes, node_label, true);
        MPI_Waitall(request_index, requests, status); // 这个地方暂时采用这种新旧混合的写法. 之后有空了再改吧.
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD);
        block_send = (block_send == 0 ? num_nodes - 1 : block_send - 1);
        block_recv = (block_recv == 0 ? num_nodes - 1 : block_recv - 1);
    }
    //LOG_IF(WARNING, node_label == 0) << "broadcast done";
    delete[] requests;
    delete[] status;
}

int main(int argc, char **argv)
{
    // 当前节点的编号, 总结点数量, 孤立节点数量
    size_t node_label, total_peers, num_lonely = 0; 

    // 命令行参数
    int repeat = 1;
    double sum_time = 0, min_time = INF;
    int comm_type = 0; // 0 for tree, 1 for ring, 2 for mpi
    bool to_file = true;

    // others
    std::vector<double> repeat_time;
    int tmp;

    // init glog
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // init mpi
    MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE, &tmp);
    CHECK_EQ(tmp, MPI_THREAD_MULTIPLE) << "MPI_THREAD_MULTIPLE thread support required " <<  tmp;
    MPI_Comm_size(MPI_COMM_WORLD, &tmp);
    total_peers = tmp;
    MPI_Comm_rank(MPI_COMM_WORLD, &tmp);
    node_label = tmp;
    LOG_IF(INFO, node_label == 0) << "glog initialized.";
    LOG(INFO) << "total " << total_peers << " and here's " << node_label;
    //if (node_label == 0) 
        google::InstallFailureSignalHandler();
    // end init

    MPI_Barrier(MPI_COMM_WORLD);
    size_t data_len = 336e3;
    std::vector<size_t> topo;
    // arg parse
    for (auto i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--size") == 0)
        {
            i++;
            CHECK_GE(argc, i);
            std::stringstream ss;
            ss << argv[i];
            ss >> data_len;
            LOG_IF(WARNING, node_label == 0) << "data size = " << data_len;
        }
        else if (strcmp(argv[i], "--repeat") == 0)
        {
            i++;
            CHECK_GE(argc, i);
            std::stringstream ss;
            ss << argv[i];
            ss >> repeat;
            LOG_IF(WARNING, node_label == 0) << "repeat = " << repeat;
        }
        else if (strcmp(argv[i], "--comm-only") == 0)
        {
            comm_only = true;
            LOG_IF(WARNING, node_label == 0) << "comm_only = true";
        }
        else if (strcmp(argv[i], "--topo") == 0)
        {
            for (i++; i < argc; i++)
            {
                std::stringstream ss;
                if (argv[i][0] == '+') // 定义孤立节点. 必须是在 --topo 参数末尾, +n 的形式.
                {
                    ss << (argv[i] + 1);
                    ss >> tmp;
                    num_lonely = tmp;
                    LOG_IF(WARNING, node_label == 0) << "Here's " << tmp << " lonely node(s)";
                    break;
                }
                ss << argv[i];
                ss >> tmp;
                if (tmp == 1)
                {
                    comm_type = 1;
                    LOG_IF(WARNING, node_label == 0) << "ring allreduce selected";
                    topo.push_back(1);
                    break;
                }
                if (tmp == 0)
                {
                    comm_type = 2;
                    LOG_IF(WARNING, node_label == 0) << "mpi allreduce selected";
                    topo.push_back(0);
                    break;
                }
                topo.push_back(tmp);
            }
            LOG_IF(WARNING, node_label == 0 && (comm_type == 0)) << "tree allreduce selected";
            break;
        }
        else if (strcmp(argv[i], "--not-to-file") == 0)
        {
            to_file = false;
            LOG_IF(WARNING, node_label == 1) << "to file = false";
        }
        else
        {
            LOG(FATAL) << "unknown parameter: " << argv[i];
        }
    }
    CHECK_NE(topo.empty(), true) << "topology should be given!";
    
    // 初始化 data 和 buffer
    DataType *data = new DataType[data_len * 2];
    recv_buffer = new DataType[data_len * 2];
    char *mpi_buffer = new char[data_len * 10];
    MPI_Buffer_attach(mpi_buffer, data_len * 10);
    for (size_t i = 0; i != data_len; i++)
    {
        data[i] = i / 1000.0;
    }
    // 准备就绪
    LOG_IF(INFO, node_label == 0) << "READY";
    if (comm_type == 0) //tree
    {
        for (auto i = 0; i != repeat; i++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto time1 = MPI_Wtime();
            tree_allreduce(data, data_len, total_peers, num_lonely, node_label, topo);
            auto time2 = MPI_Wtime();
            repeat_time.push_back(time2 - time1);
            sum_time += time2 - time1;
            min_time = std::min(time2 - time1, min_time);
            LOG_IF(WARNING, node_label == 0) << "repeat " << i << " finished"; 
        }
    }
    else if (comm_type == 1) //ring
    {
        for (auto i = 0; i != repeat; i++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto time1 = MPI_Wtime();
            ring_allreduce(data, data_len, total_peers, node_label);
            auto time2 = MPI_Wtime();
            repeat_time.push_back(time2 - time1);
            sum_time += time2 - time1;
            min_time = std::min(time2 - time1, min_time);
            LOG_IF(WARNING, node_label == 0) << "repeat " << i << " finished"; 
        }
    }
    else if (comm_type == 2) //mpi
    {
        for (auto i = 0; i != repeat; i++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto time1 = MPI_Wtime();
            MPI_Allreduce(data, recv_buffer, data_len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            auto time2 = MPI_Wtime();
            repeat_time.push_back(time2 - time1);
            sum_time += time2 - time1;
            min_time = std::min(time2 - time1, min_time);
            memcpy(data, recv_buffer, data_len * sizeof(DataType));
            LOG_IF(WARNING, node_label == 0) << "repeat " << i << " finished"; 
        }
    }
    else 
    {
        LOG(FATAL) << "unknown comm type: " << comm_type;
    }


    for (int i = 0; i != total_peers; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == node_label)
        {
            std::cout << "CHECK " << node_label << ": ";
            for (int i = 9; i != 20; i++) std::cout << data[i] << " ";
            std::cout << std::endl;
        }
    }
    
    MPI_Finalize();

    // 写入文件
    if (node_label == 0 && to_file)
    {
        std::stringstream ss;
        ss << total_peers << "." << data_len << ".";
        for (auto i : topo)
        {
            ss << i << "-";
        }
        ss << (comm_only ? ".comm_test." : ".ar_test.");
        ss << time(NULL) << ".txt";
        std::string filename;
        ss >> filename;
        write_vector_to_file(repeat_time, filename);
    }

    LOG_IF(WARNING, node_label == 0) << "DONE, average time: " << sum_time / repeat << ", min time: " << min_time << std::endl;
    google::ShutdownGoogleLogging();
    return 0;
}