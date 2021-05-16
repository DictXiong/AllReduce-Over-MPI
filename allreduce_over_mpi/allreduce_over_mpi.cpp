#include<iostream>
#include<vector>
#include<thread>
#include<mpi.h>
#include "glog/logging.h"

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

typedef float DataType;

// 全局变量, 标注当前节点的编号, 和总结点数量
int num_peer, total_peers; 

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
public:
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
        std::cerr << "Unknown num_blocks: " << num_blocks << std::endl;
        break;
    }
}

void handle_send(std::vector<Operation> *ops, DataType *data, size_t len)
{
    CHECK_NOTNULL(ops);
    CHECK_NOTNULL(data);

    size_t start;

    // 总共要进行的通信的次数
    const size_t comm_size = ops->size() * (*ops)[0].blocks.size();

    // 用于 MPI_Waitall()
    MPI_Request *request = new MPI_Request[comm_size];
    size_t request_index = 0;
    MPI_Status *status = new MPI_Status[comm_size];

    int count = len / total_peers;

    for (const auto &i : *ops)
    {
        if (LIKELY(num_peer != i.peer))
        {
            for (const auto &j : i.blocks)
            {
                start = len / total_peers * j;
                MPI_Isend(data + start, count, MPI_FLOAT, i.peer, 0, MPI_COMM_WORLD, &request[request_index++]); // 此处的tag暂时先打0
            }
        }
    }
    MPI_Waitall(request_index, request, status);
}

void handle_recv_gather(std::vector<Operation> *ops, DataType *data, size_t len)
{
    CHECK_NOTNULL(ops);
    CHECK_NOTNULL(data);

    DataType *buffer = new DataType[len]; // 创建 buffer
    size_t buffer_index = 0;
    const size_t peer_gap = (*ops)[0].blocks.size(); // buffer 中来自同一节点的数据块连续存放, 这是不同节点的相同数据块之间的间距
    const size_t count_peers = ops->size() - 1; // 要与多少人进行通信
    const size_t comm_size = count_peers * peer_gap; // 总共要进行的通信的次数

    // 用于 MPI_Waitall()
    MPI_Request *request = new MPI_Request[comm_size];
    size_t request_index = 0;
    MPI_Status *status = new MPI_Status[comm_size];

    int count = len / total_peers; // 单位数据块的大小

    for (const auto &i : *ops)
    {
        if (LIKELY(num_peer != i.peer))
        {
            for (const auto &j : i.blocks)
            {
                MPI_Irecv(buffer + buffer_index, count, MPI_FLOAT, i.peer, 0, MPI_COMM_WORLD, &request[request_index++]); // 此处的tag暂时先打0
                buffer_index += count;
            }
        }
    }

    MPI_Waitall(request_index, request, status);

    // 通信结束, 开始加和
    DataType **src = new DataType*[MAX_NUM_BLOCKS];
    for (int i = 0; i != MAX_NUM_BLOCKS; i++)
    {
        src[i] = nullptr;
    }
    for (auto i = (*ops)[0].blocks.begin(); i != (*ops)[0].blocks.end(); i++)
    {
        size_t start = len / total_peers * (*i);
        src[0] = data + start;
        start = (i - (*ops)[0].blocks.begin()) * count;
        for (size_t j = 0; j != count_peers; j++)
        {
            src[j + 1] = buffer + start;
            start += count * peer_gap;
        }
        reduce_sum(src, count_peers + 1, count);
    }
}

void handle_recv_broadcast(std::vector<Operation> *ops, DataType *data, size_t len)
{
    CHECK_NOTNULL(ops);
    CHECK_NOTNULL(data);

    size_t start;

    const size_t count_peers = ops->size() - 1; // 要与多少人进行通信
    const size_t peer_gap = (*ops)[0].blocks.size(); // 与每人通信的次数
    const size_t comm_size = count_peers * peer_gap; // 总共要进行的通信的次数

    // 用于 MPI_Waitall()
    MPI_Request *request = new MPI_Request[comm_size];
    size_t request_index = 0;
    MPI_Status *status = new MPI_Status[comm_size];

    int count = len / total_peers; // 单位数据块的大小

    for (const auto &i : *ops)
    {
        if (LIKELY(num_peer != i.peer))
        {
            for (const auto &j : i.blocks)
            {
                start = len / total_peers * j;
                MPI_Irecv(data + start, count, MPI_FLOAT, i.peer, 0, MPI_COMM_WORLD, &request[request_index++]); // 此处的tag暂时先打0
            }
        }
    }

    MPI_Waitall(request_index, request, status);
}

void all_reduce(DataType *data, size_t len, std::vector<size_t> stages)
{
    CHECK_NOTNULL(data);

    MPI_Barrier(MPI_COMM_WORLD);
    LOG_IF(WARNING, num_peer == 0) << "gathering start";
    Send_Ops send_ops(total_peers, num_peer, stages);
    Recv_Ops recv_ops(total_peers, num_peer, stages);
    send_ops.generate_ops();
    recv_ops.generate_ops();
    for (size_t i = 0; i != stages.size(); i++)
    {
        std::thread send_thread(handle_send, &(send_ops.ops[i]), data, len);
        std::thread recv_thread(handle_recv_gather, &(recv_ops.ops[i]), data, len);
        send_thread.join();
        recv_thread.join();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    LOG_IF(WARNING, num_peer == 0) << "gathering done";
    for (int i = stages.size() - 1; i >= 0; i--)
    {
        std::thread send_thread(handle_send, &(recv_ops.ops[i]), data, len);
        std::thread recv_thread(handle_recv_broadcast, &(send_ops.ops[i]), data, len);
        send_thread.join();
        recv_thread.join();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    LOG_IF(WARNING, num_peer == 0) << "broadcast done";
}

int main(int argc, char **argv)
{
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "glog initialized.";
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_peers);
    MPI_Comm_rank(MPI_COMM_WORLD, &num_peer);
    LOG(INFO) << "total " << total_peers << " and here's " << num_peer;
    CHECK_EQ(total_peers, 6);
    if (num_peer == 0) google::InstallFailureSignalHandler();
    MPI_Barrier(MPI_COMM_WORLD);

    float data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    
    all_reduce(data, 12, {2,3});

    std::cout << "summary " << num_peer << ": ";
    for (int i = 0; i != 12; i++) std::cout << data[i] << " ";
    std::cout << std::endl;

    MPI_Finalize();
    google::ShutdownGoogleLogging();
}