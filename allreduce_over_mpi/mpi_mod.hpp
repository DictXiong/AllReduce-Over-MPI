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
#include<assert.h>
#ifdef STANDALONE_TEST
#include<mpi.h>
#include<glog/logging.h>
#endif

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
// LOG 控制
//#define FT_DEBUG 
//#define SHOW_TIME // 显示更多的时间调试信息
#ifdef SHOW_TIME
double _timer_base;
#define TIMER_RESET() do {_timer_base=MPI_Wtime();} while (false)
#define TIMER_LOG_IF(exp, note) do {LOG_IF(INFO,exp)<<MPI_Wtime()-_timer_base<<" :: "<<note;} while (false)
#define TIMER_RECORD(record) do {record += (MPI_Wtime() - _timer_base);} while (false)
extern double time_reduce = 0;
#endif
// end of LOG 控制

#ifdef FT_DEBUG
#include<algorithm>
#endif

namespace FlexTree
{

bool comm_only = false; // 现在没有用, 但是以后可能会用吧. 
const int INF = 0x3F3F3F3F;
static void *recv_buffer = nullptr; //必须初始化

class Helper
{
public:
    static size_t get_split_size(size_t data_len, size_t num_split)
    {
        return (data_len + num_split - 1) / num_split;
    }
    template<class T>
    static std::string vector_to_string(std::vector<T> vec, char split = ',')
    {
        if (vec.empty()) return "";
        std::ostringstream is;
        for (const auto i:vec)
        {
            is << i << split;
        }
        auto ans = is.str();
        ans.pop_back();
        return ans;
    }
    template<class T>
    static void print_vector(std::vector<T> vec, bool newline = true, char split = ',' )
    {
        std::cout << vector_to_string(vec, split);
        if (newline)
        {
            std::cout << std::endl;
        }
    }
};

class Operation
{
public:    
    size_t peer;
    std::vector<size_t> blocks;

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

    /**
     * Operation 类构造函数. 用于 Tree AllReduce. 
     * 类似于 range(_peer % gap, gap, _total_peers)
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
    std::string to_string()const
    {
        if (blocks.empty()) return ""; 
        std::ostringstream os;
        os<< "peer " << peer <<": ";
        for (auto k:blocks)
        {
            os<<k<<",";
        }
        std::string ans(std::move(os.str()));
        ans.pop_back();
        return ans;
    }
};

// 为什么不在构造时直接使用 ft_ctx: 因为 ft_ctx 是与 mpi 强耦合的一个东西, 但是 operations 以及拓扑的生成应当只和所需要的这三个参数有关系, 不要和 mpi 扯上关系.
class Operations
{
public:
    std::vector<size_t> stages;
    size_t total_peers, node_label, num_lonely, num_split;
    bool is_lonely;
public:
    std::vector<std::vector<Operation>> ops, lonely_ops;
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
        is_lonely == (node_label >= num_split);
        size_t pi = 1;
        for (const auto &i:_stages)
        {
            pi *= i;
        }
        assert(pi == num_split);
        assert(stages.size() > 1 || num_lonely == 0);
    }
    // 生成拓扑, 要求子类实现
    virtual void generate() = 0;
    // 打印拓扑
    virtual void print()const
    {
        std::cout << "\033[0m\033[1;30;43m" << typeid(*this).name() << "\033[0m of node " << node_label << " in total " << num_split << " peers: " << std::endl;
        for (const auto &i:ops)
        {
            if (&i != &*(ops.end() - 1))
            {
                std::cout << "\033[0m\033[1;33m┝ stage\033[0m";
            }
            else 
            {
                std::cout << "\033[0m\033[1;33m┕ stage\033[0m";
            }
            for (const auto &j:i)
            {
                std::cout<< " | " << j.to_string();
            }
            std::cout<<std::endl;
        }
        //if ((node_label < num_split && has_lonely_blocks()) || (node_label >= num_split))
        if (!lonely_ops.empty())
        {
            std::cout << "AND " << num_lonely << " lonely node(s):" << std::endl;
            for (const auto &i:lonely_ops)
            {
                if (&i != &*(lonely_ops.end() - 1))
                {
                    std::cout << "\033[0m\033[1;33m┝ stage\033[0m";
                }
                else 
                {
                    std::cout << "\033[0m\033[1;33m┕ stage\033[0m";
                }
                for (const auto &j:i)
                {
                    std::cout<< " | " << j.to_string();
                }
                std::cout<<std::endl;
            }
        }
    }

    /**
     * @brief 看给的这个节点在第 h 层是否拥有 lonely blocks.
     * @param node_label (default: this->node_label)
     * @param h 高度
     * @return bool true/false
     */
    inline bool has_lonely_blocks(size_t h, size_t n = INF) const 
    {
        if (n == INF) n = node_label;
        assert(n < num_split);
        return (num_lonely > 0) && (n >= stages[0] * (num_lonely)) && (h == 0 || n % stages[0] < num_lonely);
        /**
         * 值得注意的三个条件:
         * 1. 首先, 全局得存在孤立节点, 才有这个问题.
         * 2. 其次, 该节点应当在孤立区内, 即子节点所覆盖的范围和孤立区重合.
         * 3. 最后, 如果问的是非底层的话, 那这个节点本身在最底层得负责一块孤立.
         */
    }
    /**
     * @brief 看这个孤立数据块跟着谁走
     * @param split_label
     * @return size_t 返回编号
     */
    inline size_t find_star(size_t split_label) const 
    {
        assert(split_label >= num_split);
        return split_label - stages[0];
    }
    /**
     * @brief 节点 n 不孤立且在第 h 层, 存有哪些孤立块? 
     * 
     * @param h height
     * @param n node_label, default = this->...
     * @return std::vector<size_t> 如果有, 返回孤立编号; 如果无, 返回空集
     */
    std::vector<size_t> find_followers(size_t h, size_t n = INF) const
    {
        if (n == INF) n = node_label;
        if (num_lonely == 0) return {};
        assert(n < num_split); assert(h <= stages.size() + 1);

        std::vector<size_t> ans;
        size_t gap = 1;
        
        for (int i = 0; i <= int(h) - 1; i++)
        {
            gap *= stages[i];
        }
        for (size_t i = num_split; i != total_peers; i++)
        {
            if (find_star(i) % gap == n % gap) ans.push_back(i);
        }
        if (has_lonely_blocks(h, n)) return ans;
        else return {};
    }
};

class Send_Operations: public Operations
{
public:
    using Operations::Operations;
    // 生成逻辑拓扑
    virtual void generate()
    {
        if (node_label < num_split)
        {
            // 当前组内成员的编号的间距
            size_t gap = 1;
            for (size_t i = 0; i < stages.size(); i++)
            {
                std::vector<Operation> stage_ops;
                std::vector<Operation> stage_lonely_ops;
                // 当前组内编号最小的成员
                size_t left_peer = node_label / (gap * stages[i]) * (gap * stages[i]) + node_label % gap;
                for (size_t j = 0; j < stages[i]; j++)
                {
                    stage_ops.emplace_back(left_peer, num_split, gap * stages[i]);
                    if (has_lonely_blocks(i))
                    {
                        auto followers = find_followers(i + 1, left_peer);
                        assert(followers.size() <= 1);
                        if (followers.size() == 1)
                        {
                            // 上面这堆判断逻辑链是什么意思呢? 首先, 本节点要拥有孤立块才行; 其次, 检查对方节点是否负责这一块的收集. 如果均为真, 那就传给它.
                            if (i != stages.size() - 1)
                            {
                                stage_lonely_ops.emplace_back(left_peer, followers[0]);
                            }
                            else
                            {
                                // 如果已经是最后一层了, 那么正确的操作就是传给真正负责处理这块的那个孤立节点, 而不是继续跟着走了.
                                stage_lonely_ops.emplace_back(followers[0], followers[0]);
                            }
                        }
                    }
                    left_peer += gap;
                }
                if (i == 0 && num_lonely > 0 && node_label < stages[0] * num_lonely)
                {
                    /**
                     * 又来讲讲这个判断是个啥. 这里是为了处理第一步中的扩展组, 需要从孤立节点处收集非孤立块. 需要判断:
                     * 1. 得是第一步
                     * 2. 确实存在孤立节点
                     * 3. 这个点确实属于扩展组
                     */ 
                    Operation tmp_op(num_split + node_label / stages[0], num_split);
                    for (size_t i = num_split + 1; i < total_peers; i++)
                    {
                        tmp_op.blocks.push_back(i);
                    }
                    stage_ops.push_back(tmp_op);
                }
                ops.push_back(stage_ops);
                lonely_ops.push_back(stage_lonely_ops);
                gap *= stages[i];
            }
        }
        else
        {
            // 打起精神! 现在我是孤立节点!
            // 第一步
            {            
                std::vector<Operation> stage_lonely_ops;
                size_t left_peer = (node_label - num_split) * stages[0];
                for (size_t i = 0; i < stages[0]; i++)
                {
                    stage_lonely_ops.emplace_back(left_peer + i, num_split, stages[0]);
                }
                lonely_ops.push_back(stage_lonely_ops);
            }
            // 中间步骤
            {
                std::vector<Operation> stage_lonely_ops;
                for (size_t i = num_split; i < total_peers; i++)
                {
                    stage_lonely_ops.emplace_back(i, i);
                }
                lonely_ops.push_back(stage_lonely_ops);
            }
            // 最后一步不发东西. 然后需要进行对齐.
            for (size_t i = 2; i < stages.size(); i++)
            {
                lonely_ops.emplace_back();
            }
        }
    }
};

class Recv_Operations: public Operations
{
public:
    using Operations::Operations;
    // 生成逻辑拓扑
    virtual void generate()
    {
        if (node_label < num_split)
        {
            // 当前组内成员的编号的间距
            size_t gap = 1;
            for (size_t i = 0; i < stages.size(); i++)
            {
                std::vector<Operation> stage_ops;
                Operation op_template(node_label, num_split, gap * stages[i]);
                std::vector<Operation> stage_lonely_ops;
                auto followers = find_followers(i + 1);
                assert(followers.size() <= 1);

                // 当前组内编号最小的成员
                size_t left_peer = node_label / (gap * stages[i]) * (gap * stages[i]) + node_label % gap;

                for (size_t j = 0; j < stages[i]; j++)
                {
                    op_template.peer = left_peer;
                    stage_ops.emplace_back(op_template);
                    if (followers.size() > 0 && has_lonely_blocks(i, left_peer) && i != stages.size() - 1)
                    {
                        /**
                         * 讲讲这个复杂的判断条件. 如果要加之入这一步的接收操作, 需要同时满足
                         * 1. 得是本节点确实要负责这块的收集
                         * 2. 然后, 对方得确实拥有这块孤立块
                         * 3. 最后, 不能是最后一步, 因为最后一步直接传给孤立节点
                         */ 
                        stage_lonely_ops.emplace_back(left_peer, followers[0]);
                    }
                    left_peer += gap;
                }
                if (i == 0 && num_lonely > 0 && node_label < stages[0] * num_lonely)
                {
                    /**
                     * 又来讲讲这个判断是个啥. 这里是为了处理第一步中的扩展组, 需要从孤立节点处收集非孤立块. 需要判断:
                     * 1. 得是第一步
                     * 2. 确实存在孤立节点
                     * 3. 这个点确实属于扩展组
                     */ 
                    op_template.peer = num_split + node_label / stages[0];
                    stage_ops.emplace_back(op_template);
                }
                ops.push_back(stage_ops);
                lonely_ops.push_back(stage_lonely_ops);
                gap *= stages[i];
            }
        }
        else
        {
            // 好了, 百年孤独.
            // 第一步
            {
                std::vector<Operation> stage_lonely_ops;
                Operation lonely_op_template(node_label, num_split);
                for (size_t i = num_split + 1; i < total_peers; i++)
                {
                    lonely_op_template.blocks.push_back(i);
                }
                size_t left_peer = (node_label - num_split) * stages[0];
                for (size_t i = 0; i < stages[0]; i++)
                {
                    lonely_op_template.peer = left_peer + i;
                    stage_lonely_ops.emplace_back(lonely_op_template);
                }
                lonely_ops.push_back(stage_lonely_ops);
            }
            // 中间步骤
            {
                std::vector<Operation> stage_lonely_ops;
                for (size_t i = num_split; i < total_peers; i++)
                {
                    stage_lonely_ops.emplace_back(i, node_label);
                }
                lonely_ops.push_back(stage_lonely_ops);
            }
            // 先进行对齐
            for (size_t i = 2; i < stages.size(); i++)
            {
                lonely_ops.emplace_back();
            }
            // 最后一步要从不孤独的人们那儿收点税
            {
                auto stage_lonely_ops = lonely_ops.end() - 1;
                size_t gap = num_split / *(stages.end() - 1);
                for (int i = node_label - stages[0]; i >= 0; i -= gap)
                {
                    if (find_followers(stages.size() - 1, i).size() == 1)
                    {
                        assert(find_followers(stages.size() - 1, i)[0] == node_label);
                        stage_lonely_ops->emplace_back(i, node_label);
                    }
                }
            }
        }
    }
};

/**
 * FMA = FlexTree-MPI Adapter
 * 用这种缩写看起来就很高大上
 * 其实很简单: 就是标明向谁发哪块内存地址的数据
 * 比 Operation 更近了一步. 不过可以认为是 Operation 推理细化的结果.
 */
class FMA_Operation
{
public:
    struct _Memory_Range
    {
        /**
         * @brief 简单结构体. addr 是起始地址, len 是长度.
         * actual_addr 是: 这块数据, 本来在 src 中偏移量是多少? 用于 reduce 的时候能找到 reduce 后的数据往哪儿放.
         */
        size_t addr, len, actual_addr;
        _Memory_Range()
        {
            len = 0;
        } 
        _Memory_Range(size_t _addr, size_t _len, size_t _actual_addr): addr(_addr), len(_len), actual_addr(_actual_addr)
        {}
        std::string to_string()const 
        {
            std::ostringstream os;
            os << addr;
            if (addr != actual_addr) os << "(" << actual_addr << ")";
            os << "+" << len;
            return os.str();
        }
    };

public:
    size_t peer;
    std::vector<_Memory_Range> ranges;
    bool valid = false, from_src = false;
    /**
     * from_src 是指, 是从 src 里面取 (false), 还是从 dst 里面取数据 (true)
     * 这里展开讲讲:
     * 1. 对于发送操作来说, from_src matters. 因为发送要么是从 src 发送, 要么是从 dst 发送. 这个 flag 可以在发送的时候指导从哪儿取数据.
     * 2. 对于接受操作来说, it doesn't matter. 因为无论如何肯定是接收到 buffer 里面的.
     */
public:
    FMA_Operation()
    {
        valid = false;
    }
    FMA_Operation(size_t _peer, bool _from_src):peer(_peer), from_src(_from_src)
    {}
    /**
     * @brief Construct a new fma operation object
     * 根据 peer 和 block 来构造. 
     * 如果给出了 _addr, 那就修改源地址. 一般配合 from_src = false 使用, 考虑接收的情况.
     * 
     * @param _peer 
     * @param block_label 
     * @param num_nodes 总共的节点数, 也就是切分块数
     * @param data_len 
     * @param _from_src from_src 是指, 是从源数据块里面取 (true), 还是从 dst 里面取数据 (false)
     */
    FMA_Operation(size_t _peer, size_t block_label, size_t num_nodes, size_t data_len, bool _from_src, size_t _addr = INF): peer(_peer), from_src(_from_src)
    {
        valid = true;
        assert(peer < num_nodes);

        push_block_back(block_label, num_nodes, data_len, _addr);
    }
    void push_block_back(size_t block_label, size_t num_nodes, size_t data_len, size_t _addr = INF)
    {
        assert(block_label < num_nodes);
        assert(peer < num_nodes);


        size_t split_size = Helper::get_split_size(data_len, num_nodes);
        size_t actual_addr = split_size * block_label;
        size_t len, addr;
        if (actual_addr > data_len)
        {
            len = 0;
        }
        else if (actual_addr + split_size > data_len)
        {
            len = data_len - actual_addr;
        }
        else 
        {
            len = split_size;
        }
        if (_addr == INF)
        {
            addr = actual_addr;
        }
        else 
        {
            addr = _addr;
        }
        ranges.emplace_back(addr, len, actual_addr);
    }
    std::string to_string()const
    {
        if (ranges.empty()) return "";
        std::ostringstream os;
        os << "peer " << peer << (from_src ? 's' : 'n') << ": ";
        for (const auto &i:ranges)
        {
            os << i.to_string() << ",";
        }
        std::string ans(std::move(os.str()));
        ans.pop_back();
        return ans;
    }
};

class FMA_Operations
{
public:
    Send_Operations *raw_send_operations = nullptr;
    Recv_Operations *raw_recv_operations = nullptr;
    std::vector<std::vector<FMA_Operation>> FMA_ops, FMA_lonely_ops;
    size_t num_nodes, data_len;
    FMA_Operations()
    {
        raw_send_operations = nullptr;
        raw_recv_operations = nullptr;
    }
    FMA_Operations(Send_Operations* sops, Recv_Operations* rops, size_t _num_nodes, size_t _data_len):num_nodes(_num_nodes), data_len(_data_len) 
    {
        assert(sops != nullptr && rops != nullptr);
        raw_send_operations = sops;
        raw_recv_operations = rops;
    }
    virtual void generate() = 0;
    virtual void print()const
    {
        std::cout << "\033[0m\033[1;30;42m" << typeid(*this).name() << "\033[0m" << " of node " << raw_send_operations->node_label << " in total " << raw_send_operations->num_split << " peers: " << std::endl;
        for (const auto &i:FMA_ops)
        {
            if (&i != &*(FMA_ops.end() - 1))
            {
                std::cout << "\033[0m\033[1;32m┝ stage\033[0m";
            }
            else 
            {
                std::cout << "\033[0m\033[1;32m┕ stage\033[0m";
            }
            for (const auto &j:i)
            {
                std::cout<< " | " << j.to_string();
            }
            std::cout<<std::endl;
        }
        if (!FMA_lonely_ops.empty())
        {
            std::cout << "AND lonely op(s):" << std::endl;
            for (const auto &i:FMA_lonely_ops)
            {
                if (&i != &*(FMA_lonely_ops.end() - 1))
                {
                    std::cout << "\033[0m\033[1;32m┝ stage\033[0m";
                }
                else 
                {
                    std::cout << "\033[0m\033[1;32m┕ stage\033[0m";
                }
                for (const auto &j:i)
                {
                    std::cout<< " | " << j.to_string();
                }
                std::cout<<std::endl;
            }
        }
    }
};

class FMA_Send_Operations : public FMA_Operations
{
public:
    using FMA_Operations::FMA_Operations;
    /**
     * @brief 生成该节点整个生命周期中所有的发送操作
     * 
     */
    virtual void generate()
    {
        assert(raw_send_operations != nullptr && raw_recv_operations != nullptr);

        auto generate_from_ops = [&](std::vector<FMA_Operation> *stage_ops, std::vector<Operation> *ops, bool is_src)
        {
            for (const auto &j: *ops)
            {
                FMA_Operation tmp_op(j.peer, is_src);
                for (const auto &k : j.blocks)
                {
                    tmp_op.push_block_back(k, num_nodes, data_len);
                }
                stage_ops->push_back(tmp_op);
            }
        };
        const size_t num_stages = raw_send_operations->stages.size();
        if (!raw_send_operations->ops.empty())
        {
            for (size_t i = 0; i < num_stages; i++)
            {
                std::vector<FMA_Operation> stage_ops;
                generate_from_ops(&stage_ops, &raw_send_operations->ops[i], (i == 0));
                FMA_ops.push_back(stage_ops);
            }
        }
        if (!raw_recv_operations->ops.empty())
        {
            for (int i = num_stages - 1; i >= 0; i--)
            {
                std::vector<FMA_Operation> stage_ops;
                generate_from_ops(&stage_ops, &raw_recv_operations->ops[i], false);
                FMA_ops.push_back(stage_ops);
            }
        }
        // 孤立节点
        if (!raw_send_operations->lonely_ops.empty())
        {
            for (size_t i = 0; i < num_stages; i++)
            {
                std::vector<FMA_Operation> stage_ops;
                generate_from_ops(&stage_ops, &raw_send_operations->lonely_ops[i], (i == 0));
                FMA_lonely_ops.push_back(stage_ops);
            }
        }
        if (!raw_recv_operations->lonely_ops.empty())
        {
            for (int i = num_stages - 1; i >= 0; i--)
            {
                std::vector<FMA_Operation> stage_ops;
                generate_from_ops(&stage_ops, &raw_recv_operations->lonely_ops[i], false);
                FMA_lonely_ops.push_back(stage_ops);
            }
        }
    }
};

class FMA_Recv_Operations : public FMA_Operations
{
public:
    using FMA_Operations::FMA_Operations;
    /**
     * @brief 生成该节点整个生命周期中所有的接收操作
     */
    virtual void generate()
    {
        assert(raw_send_operations != nullptr && raw_recv_operations != nullptr);
        size_t split_size = Helper::get_split_size(data_len, num_nodes);
        auto generate_from_ops = [&](std::vector<FMA_Operation> *stage_ops, std::vector<Operation> *ops, bool accordingly, size_t offset = 0)
        {
            for (const auto &j: *ops)
            {
                FMA_Operation tmp_op(j.peer, false);
                for (const auto &k : j.blocks)
                {
                    if (!accordingly)
                    {
                        // 则平铺直叙
                        tmp_op.push_block_back(k, num_nodes, data_len, offset);
                        (offset) += split_size;
                    }
                    else
                    {
                        // 则对位覆写
                        tmp_op.push_block_back(k, num_nodes, data_len);
                    }
                }
                stage_ops->push_back(std::move(tmp_op)); // ?
            }
        };

        const size_t num_stages = raw_send_operations->stages.size();
        if (!raw_recv_operations->ops.empty())
        {
            for (size_t i = 0; i < num_stages; i++)
            {
                std::vector<FMA_Operation> stage_ops;
                generate_from_ops(&stage_ops, &raw_recv_operations->ops[i], false);
                FMA_ops.push_back(stage_ops);
            }
        }
        if (!raw_send_operations->ops.empty())
        {
            for (int i = num_stages - 1; i >= 0; i--)
            {
                std::vector<FMA_Operation> stage_ops;
                generate_from_ops(&stage_ops, &raw_send_operations->ops[i], true);
                FMA_ops.push_back(stage_ops);
            }
        }
        // 孤立节点
        size_t offset = split_size * num_nodes;
        if (!raw_recv_operations->lonely_ops.empty())
        {
            for (size_t i = 0; i < num_stages; i++)
            {
                std::vector<FMA_Operation> stage_ops;
                generate_from_ops(&stage_ops, &raw_recv_operations->lonely_ops[i], false, offset);
                FMA_lonely_ops.push_back(stage_ops);
            }
        }
        if (!raw_send_operations->lonely_ops.empty())
        {
            for (int i = num_stages - 1; i >= 0; i--)
            {
                std::vector<FMA_Operation> stage_ops;
                generate_from_ops(&stage_ops, &raw_send_operations->lonely_ops[i], true);
                FMA_lonely_ops.push_back(stage_ops);
            }
        }
    }
};

/**
 * @brief FlexTree 的各种上下文, 反正挺管用的一个东西, 啥都往里放.
 * 要注意的是: num_split 的这个定义 #nodes - #lonely 其实是一个历史遗留问题. 
 * 在之前, +n 的时候, #split 只和非孤立节点数相关; 但是现在的设计是无论如何都
 * 分成 #nodes 份. 很多地方都使用了 num_split, 所以这个地方保留吧. 之后也许
 * 可以考虑改名.
 * 
 */
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
    static size_t get_num_nodes(const MPI_Comm &_comm)
    {
        int tmp;
        MPI_Comm_size(_comm, &tmp);
        return tmp;
    }
};

const size_t MAX_NUM_BLOCKS = 20;
template<class DataType> 
static void reduce_sum(const DataType **src, DataType *dst, const int &num_blocks, const size_t &num_elements)
{
#ifdef FT_DEBUG
    //std::cout << "reduce_sum called, ele size = " << sizeof(**src) << ", num_blocks = " << num_blocks << ", num_ele = " << num_elements <<  std::endl;
    //std::cout << "AND: " << src[0][0] << " " << src[1][0] << " " << dst[0] << std::endl;
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
    case 1:
    {
        if (dst != src0)
        {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
            for (size_t i = 0; i < num_elements; ++i)
            {
                dst[i] = src0[i];
            }
        }
        break;
    }
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
    std::cout << "reduce_band called, ele size = " << sizeof(**src) << std::endl;
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
    case 1:
    {
        if (dst != src0)
        {
#pragma omp parallel for simd num_threads(PARALLEL_THREAD)
            for (size_t i = 0; i < num_elements; ++i)
            {
                dst[i] = src0[i];
            }
        }
        break;
    }
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
static size_t handle_send(const MPI_Comm &comm, const MPI_Datatype &datatype, const std::vector<FMA_Operation> *fma_ops, const void *data, const FlexTree_Context &ft_ctx, MPI_Request request[])
{
    size_t request_index = 0;
    //std::cout << "ALL SAYS: handle_send Here's " << ft_ctx.node_label << " and I'm here." << std::endl; 

    for (const auto &i : *fma_ops)
    {
        if (LIKELY(ft_ctx.node_label != i.peer))
        {
            for (const auto &j : i.ranges)
            {
#ifdef FT_DEBUG
                if (ft_ctx.node_label == 8 || false) std::cout << ft_ctx.node_label << " send " << j.to_string() << " to " << i.peer << ", element size = " << ft_ctx.type_size << std::endl;
#endif
                if (j.len > 0)
                {
                    MPI_Isend(data + j.addr * ft_ctx.type_size, j.len, datatype, i.peer, 0, comm, &request[request_index++]); // 此处的tag暂时先打0
                }
            }
        }
    }
    return request_index;
}

// 同上, 只负责安排工作, 不等待工作完成.
// accordingly 参数的含义是, 如果为 true, 那么把数据块写到 buffer 中对应的位置去; 如果为 false, 那么直接平铺在 buffer 中.
// 另: 新版本将内存区域计算移动到了 FMA 中, 所以 accordingly 参数被弃用. 公示 (2021-12-02)
static size_t handle_recv(const MPI_Comm &comm, const MPI_Datatype &datatype, const std::vector<FMA_Operation> *fma_ops, void *buffer, const FlexTree_Context &ft_ctx, MPI_Request request[])
{
    size_t request_index = 0;

    for (const auto &i : *fma_ops)
    {
        if (LIKELY(ft_ctx.node_label != i.peer))
        {
            for (const auto &j : i.ranges)
            {
                #ifdef FT_DEBUG
                if (ft_ctx.node_label == 8) std::cout << ft_ctx.node_label << " recv " << j.to_string() << " from " << i.peer << ", element size = " << ft_ctx.type_size << std::endl;
                #endif
                if (j.len > 0)
                {
                    MPI_Irecv(buffer + j.addr * ft_ctx.type_size, j.len, datatype, i.peer, 0, comm, &request[request_index++]); // 此处的tag暂时先打0
                }
            }
        }
    }
    #ifdef FT_DEBUG
    if(ft_ctx.node_label == 0) std::cout << "REQ INDEX = " << request_index << std::endl;
    #endif
    return request_index;
}


/**
 * @brief 负责进行加和, 然后放到指定的位置上去. 注意会自动包含自己的那块data.
 * 这里的 dest 是一块和 data 大小/结构相同的一块内存. 进行 reduce 的时候, 会把结果对应地放进 dest 去. 注意 dest 不可以是 null.
 * 讲一讲 data, dest, buffer 三者的区别:
 * 1. data 是存放有效数据的地方, 有可能是最初的 src, 也有可能是聚合过程中的 dst. 这里主要用于取出该节点自己的这一块数据来进行加和.
 * 2. dest 是最终加和完了的东西会放过去的地方. 注意当然不是直接放在 dest 的位置, 这里会放到这一块本身对应的偏移量那儿去.
 * 3. buffer, 顾名思义, 就是 recv 放的这块内存, 存有从隔离王叔叔那儿取来的未处理的数据.
 */
static void handle_reduce(const MPI_Datatype &datatype, const MPI_Op &op, const std::vector<FMA_Operation> *fma_ops, void *buffer, const void *data, void *dest, const FlexTree_Context &ft_ctx)
{
    if (dest == nullptr)
    {
        std::cerr << "I can't reduce to null. Aborted." << std::endl;
        exit(1);
    }
    const void **src = (const void**)(new char*[fma_ops->size() + 10]);
    // 注意: 这个 #blocks 是指, 从每个人那儿, 收几块. 和下面调用 reduce_xxx 的 #blocks 以及 block_index 不是一个概念, 那个其实是有几个人, 是reduce_xxx 里面的 'block' 的概念. 
    const size_t num_blocks = fma_ops->begin()->ranges.size();
    for (int i = 0; i < num_blocks; i++)
    {
        void *dst = nullptr;
        const size_t block_len = fma_ops->begin()->ranges[i].len;
        if (block_len == 0)
        {
            #ifdef FT_DEBUG
            if (ft_ctx.node_label == 0) std::cout << ft_ctx.node_label << " will not reduce " << fma_ops->begin()->ranges[i].to_string() << " because it's empty." << std::endl;
            #endif
            continue; // 当前块实际大小为零, 直接溜了.
        }
        size_t block_index = 0;
        { // 自己身上的这块
            src[block_index++] = data + fma_ops->begin()->ranges[i].actual_addr * ft_ctx.type_size;
            dst = dest + fma_ops->begin()->ranges[i].actual_addr * ft_ctx.type_size;
            #ifdef FT_DEBUG
            if (ft_ctx.node_label == 8) std::cout << "WATCH == :" << ((float*)src[block_index-1])[2] << std::endl;
            #endif
        }
        for (const auto &j : *fma_ops)
        {
            #ifdef FT_DEBUG
            if (ft_ctx.node_label == 8) std::cout << ft_ctx.node_label << " will reduce " << j.ranges[i].to_string() << " from " << j.peer << ", element size = " << ft_ctx.type_size << std::endl;
            #endif
            if (LIKELY(j.peer != ft_ctx.node_label))
            {
                src[block_index++] = buffer + j.ranges[i].addr * ft_ctx.type_size;
                #ifdef FT_DEBUG
                if (ft_ctx.node_label == 8) std::cout << "WATCH != :" << ((float*)src[block_index-1])[2] << std::endl;
                #endif
            }
        }

        assert(dst != nullptr); // 理论上来说上面的 else 至少被执行一次. 如果没有被执行, 就是我写出 bug 来了.
        #ifdef FT_DEBUG
        if (ft_ctx.node_label == 8) if (ft_ctx.node_label == 8) std::cout << ft_ctx.node_label << " call reduce_sum, #blocks = " << std::endl;
        #endif
        if (op == MPI_SUM)
        {
            if (datatype == MPI_UINT8_T) reduce_sum((const uint8_t**)src, (uint8_t*)dst, block_index, block_len);
            else if (datatype == MPI_INT8_T) reduce_sum((const int8_t**)src, (int8_t*)dst, block_index, block_len);
            else if (datatype == MPI_UINT16_T) reduce_sum((const uint16_t**)src, (uint16_t*)dst, block_index, block_len);
            else if (datatype == MPI_INT16_T) reduce_sum((const int16_t**)src, (int16_t*)dst, block_index, block_len);
            else if (datatype == MPI_INT32_T) reduce_sum((const int32_t**)src, (int32_t*)dst, block_index, block_len);
            else if (datatype == MPI_INT64_T) reduce_sum((const int64_t**)src, (int64_t*)dst, block_index, block_len);
            else if (datatype == MPI_FLOAT) reduce_sum((const float**)src, (float*)dst, block_index, block_len);
            else if (datatype == MPI_DOUBLE) reduce_sum((const double**)src, (double*)dst, block_index, block_len);
            else if (datatype == MPI_C_BOOL) reduce_sum((const bool**)src, (bool*)dst, block_index, block_len);
            else if (datatype == MPI_LONG_LONG_INT) reduce_sum((const long long int**)src, (long long int*)dst, block_index, block_len);
            else if (datatype == MPI_LONG_LONG) reduce_sum((const long long**)src, (long long*)dst, block_index, block_len);
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
            if (datatype == MPI_UINT8_T) reduce_band((const uint8_t**)src, (uint8_t*)dst, block_index, block_len);
            else if (datatype == MPI_INT8_T) reduce_band((const int8_t**)src, (int8_t*)dst, block_index, block_len);
            else if (datatype == MPI_UINT16_T) reduce_band((const uint16_t**)src, (uint16_t*)dst, block_index, block_len);
            else if (datatype == MPI_INT16_T) reduce_band((const int16_t**)src, (int16_t*)dst, block_index, block_len);
            else if (datatype == MPI_INT32_T) reduce_band((const int32_t**)src, (int32_t*)dst, block_index, block_len);
            else if (datatype == MPI_INT64_T) reduce_band((const int64_t**)src, (int64_t*)dst, block_index, block_len);
            else if (datatype == MPI_LONG_LONG_INT) reduce_band((const long long int**)src, (long long int*)dst, block_index, block_len);
            else if (datatype == MPI_LONG_LONG) reduce_band((const long long**)src, (long long*)dst, block_index, block_len);
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
}

// 从环境变量获取每一层宽度
// 任意一个位置是 1, 那就用 ring
static std::pair<std::vector<size_t>, size_t> get_stages(const size_t &num_nodes)
{
    
    std::vector<size_t> ans;
    size_t num_lonely = 0;
    int tmp;

    //lonely
    auto FT_LONELY_raw = getenv("FT_LONELY");
    std::string FT_LONELY;
    if (FT_LONELY_raw != nullptr)
    {
        FT_LONELY = FT_LONELY_raw;
    }
    if (!FT_LONELY.empty())
    {
        std::istringstream is(FT_LONELY);
        is >> num_lonely;
    }

    //not lonely
    auto FT_TOPO_raw = getenv("FT_TOPO");
    std::string FT_TOPO; 
    size_t pi = 1;
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
            if (tmp == 1)
            {
                return std::make_pair((std::vector<size_t>){1}, 0);
            }
            ans.push_back(tmp);
            pi *= tmp;
        }
    }

    // check
    if (pi + num_lonely != num_nodes || (num_lonely != 0 && ans.size() < 2))
    {
        std::cerr << "invalid FT_TOPO " << FT_TOPO << std::endl;
        exit(1);
    }
    #ifdef FT_DEBUG
    std::cout << "FlexTree topo is ";
    for (auto i:ans)
    {
        std::cout << i << " ";
    }
    if (num_lonely != 0) std::cout << "+" << num_lonely;
    std::cout << std::endl;
    #endif
    return std::make_pair(ans, num_lonely);
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

// 如果需要原地 ar, 那么将 data 置为 nullptr.
static void tree_allreduce(const MPI_Datatype &datatype, const MPI_Op &op, const MPI_Comm &comm, const void *data, void *dst, const FlexTree_Context &ft_ctx, const std::vector<size_t> &stages)
{
#ifdef FT_DEBUG
    //std::cout << "FT DEBUG: inside treeallre: op " << op << "; len = " << len << "; total = " << num_nodes << "; datatype = " << datatype << std::endl;
#endif
    if (data == nullptr)
    {
        data = dst;
    }
    //LOG_IF(WARNING, node_label == 0) << "gathering start";
    Send_Operations send_ops(ft_ctx.num_nodes, ft_ctx.num_lonely, ft_ctx.node_label, stages);
    Recv_Operations recv_ops(ft_ctx.num_nodes, ft_ctx.num_lonely, ft_ctx.node_label, stages);
    send_ops.generate();
    recv_ops.generate();
    FMA_Send_Operations fma_send_ops(&send_ops, &recv_ops, ft_ctx.num_nodes, ft_ctx.data_size);
    FMA_Recv_Operations fma_recv_ops(&send_ops, &recv_ops, ft_ctx.num_nodes, ft_ctx.data_size);
    fma_send_ops.generate();
    fma_recv_ops.generate();

    MPI_Comm sub_comm = comm;
    const size_t MAX_COMM_SIZE = 2 * (ft_ctx.num_split - 1) * (ft_ctx.num_split);
    size_t request_index = 0;
    MPI_Request *requests = new MPI_Request[MAX_COMM_SIZE];
    MPI_Status *status = new MPI_Status[MAX_COMM_SIZE];
    //size_t lonely_request_index = 0;
    //MPI_Request *lonely_requests;
    int tmp;
    const size_t num_stages = stages.size();
    if (ft_ctx.node_label < ft_ctx.num_split)
    {
        MPI_Comm_split(comm, 0, ft_ctx.node_label, &sub_comm);
    }
    else 
    {
        MPI_Comm_split(comm, 1, ft_ctx.node_label, &sub_comm);
    }
    // UPDATE: 重大更新, 我发现孤立节点和常规节点基本上可以统一处理了, 不同分支. 2021-12-02
    for (size_t i = 0; i < num_stages; i++)
    {
        #ifdef FT_DEBUG
        std::cout << "ALL SAYS: Here's " << ft_ctx.node_label << " and I'm here." << std::endl; 
        #endif
        // 我现在就在想: fma_send_ops.FMA_ops[i].begin()->from_src 是不是全等于 i == 0? 或许有空可以 assert 一下.
        request_index = 0;
        if (!fma_send_ops.FMA_ops.empty())
        {
            request_index += handle_send(comm, datatype, &(fma_send_ops.FMA_ops[i]), fma_send_ops.FMA_ops[i].begin()->from_src ? data : dst, ft_ctx, requests + request_index);
        }
        // 另: 能不能想办法把这些 if 去掉? 或许可以在 handle_** 里面加一些判断?
        if (!fma_send_ops.FMA_lonely_ops[i].empty())
        {
            request_index += handle_send(comm, datatype,  &(fma_send_ops.FMA_lonely_ops[i]), fma_send_ops.FMA_lonely_ops[i].begin()->from_src ? data : dst, ft_ctx, requests + request_index);
        }
        tmp = 0;
        if (!fma_recv_ops.FMA_ops.empty())
        {
            tmp += handle_recv(comm, datatype, &(fma_recv_ops.FMA_ops[i]), recv_buffer, ft_ctx, requests + request_index);
        }
        if (!fma_recv_ops.FMA_lonely_ops[i].empty())
        {
            tmp += handle_recv(comm, datatype, &(fma_recv_ops.FMA_lonely_ops[i]), recv_buffer, ft_ctx, requests + request_index + tmp);
        }
        // 先等接收完毕. 因为一旦接收完毕, 就可以开始计算 (reduce) 了.
        MPI_Waitall(tmp, requests + request_index, status);
        #ifdef FT_DEBUG
        if(ft_ctx.node_label == 0) std::cout << "HERE AND: " << tmp << ", recv_buffer[15] = " << ((float*)(recv_buffer))[15] << std::endl;
        #endif
        #ifdef SHOW_TIME
            TIMER_RESET();
        #endif
        if (!fma_recv_ops.FMA_ops.empty())
        {
            handle_reduce(datatype, op, &(fma_recv_ops.FMA_ops[i]), recv_buffer, (i == 0 ? data : dst), dst, ft_ctx);
        }
        if (!fma_recv_ops.FMA_lonely_ops[i].empty())
        {
            handle_reduce(datatype, op, &(fma_recv_ops.FMA_lonely_ops[i]), recv_buffer, (i == 0 ? data : dst), dst, ft_ctx);
        }
        #ifdef SHOW_TIME
            TIMER_RECORD(time_reduce);
        #endif
        #ifdef FT_DEBUG
        if (ft_ctx.node_label == 0)
        {
            std::cout << "After reduce: dst[3] = " << ((float*)dst)[3] << std::endl; 
        }
        #endif
        MPI_Waitall(request_index, requests, status);
        MPI_Barrier(i == 0 || i == num_stages - 1 ? comm : sub_comm);
    }
    //LOG_IF(WARNING, node_label == 0) << "gathering done";
    #ifdef FT_DEBUG
    for (int i = 0; i <= ft_ctx.num_nodes; i++)
    {
        MPI_Barrier(comm);
        if (i == ft_ctx.node_label + 1)
        {
            std::cout << "Med CHECK " << ft_ctx.node_label << ": ";
            for (int i = 0; i < std::min(100UL, ft_ctx.data_size); i++) std::cout << ((float*)dst)[i] << " ";
            std::cout << std::endl;
        }
        MPI_Barrier(comm);
    }
    if (ft_ctx.node_label == 0) std::cout << "-------- FT DEBUG: complete reduce --------" << std::endl;
    #endif
    for (int i = num_stages; i < (num_stages << 1); i++)
    {
        //if (ft_ctx.node_label == 0) std::cout << "***FUCK: " << i << std::endl;
        request_index = 0;
        if (!fma_send_ops.FMA_ops.empty())
        {
            request_index += handle_send(comm, datatype, &(fma_send_ops.FMA_ops[i]), dst, ft_ctx, requests);
        }
        //if (ft_ctx.node_label == 0) std::cout << "***FUCK: " << fma_send_ops.FMA_ops[i].begin()->to_string() << std::endl;
        if (!fma_send_ops.FMA_lonely_ops[i].empty())
        {
            request_index += handle_send(comm, datatype, &(fma_send_ops.FMA_lonely_ops[i]), dst, ft_ctx, requests + request_index);
        }
        if (!fma_recv_ops.FMA_ops.empty())
        {
            request_index += handle_recv(comm, datatype, &(fma_recv_ops.FMA_ops[i]), dst, ft_ctx, requests + request_index);
        }
        //if (ft_ctx.node_label == 0) std::cout << "***FUCK: " << fma_recv_ops.FMA_ops[i].begin()->to_string() << std::endl;
        if (!fma_recv_ops.FMA_lonely_ops[i].empty())
        {
            request_index += handle_recv(comm, datatype, &(fma_recv_ops.FMA_lonely_ops[i]), dst, ft_ctx, requests + request_index);
        }
        MPI_Waitall(request_index, requests, status);
        MPI_Barrier(i == num_stages || i == (num_stages << 1) - 1 ? comm : sub_comm);
    }

    delete[] requests;
    delete[] status;
    requests = nullptr;
    status = nullptr;
    #ifdef FT_DEBUG
    if (ft_ctx.node_label == 0) std::cout << "-------- FT DEBUG: complete allreduce --------" << std::endl;
    #endif
    //LOG_IF(WARNING, node_label == 0) << "broadcast done";
    #ifdef FT_DEBUG
    for (int i = 0; i <= ft_ctx.num_nodes; i++)
    {
        MPI_Barrier(comm);
        if (i == ft_ctx.node_label + 1)
        {
            std::cout << "Tail CHECK " << ft_ctx.node_label << ": ";
            for (int i = 0; i < std::min(100UL, ft_ctx.data_size); i++) std::cout << ((float*)dst)[i] << " ";
            std::cout << std::endl;
        }
        MPI_Barrier(comm);
    }
    //std::cout << "WHY HERE: " << ((int32_t*)dst)[9] << " " << ((int32_t*)dst)[12] << std::endl;
    #endif
}

static void ring_allreduce(const MPI_Datatype &datatype, const MPI_Op &op, const MPI_Comm &comm, const void *data, void *dst, const FlexTree_Context &ft_ctx)
{
    if (data == nullptr)
    {
        data = dst;
    }
    const size_t left = (ft_ctx.node_label == 0 ? ft_ctx.num_nodes - 1 : ft_ctx.node_label - 1);
    const size_t right = (ft_ctx.node_label == ft_ctx.num_nodes - 1 ? 0 : ft_ctx.node_label + 1);
    size_t block_send = ft_ctx.node_label;
    size_t block_recv = left;
    const size_t MAX_COMM_SIZE = 4;
    size_t request_index = 0;
    MPI_Request *requests = new MPI_Request[MAX_COMM_SIZE];
    MPI_Status *status = new MPI_Status[MAX_COMM_SIZE];
    
    //LOG_IF(WARNING, node_label == 0) << "gathering start";
    for (size_t i = 0; i != ft_ctx.num_nodes - 1; i++)
    {
        std::vector<FMA_Operation> fma_send_ops = {FMA_Operation(right, block_send, ft_ctx.num_nodes, ft_ctx.data_size, i == 0)};
        std::vector<FMA_Operation> fma_recv_ops = {
            FMA_Operation(left, block_recv, ft_ctx.num_nodes, ft_ctx.data_size, false), 
            FMA_Operation(ft_ctx.node_label, block_recv, ft_ctx.num_nodes, ft_ctx.data_size, false)
        };
        request_index = handle_send(comm, datatype, &fma_send_ops, (i == 0 ? data : dst), ft_ctx, requests);
        request_index += handle_recv(comm, datatype, &fma_recv_ops, recv_buffer, ft_ctx, requests + request_index);
        MPI_Waitall(request_index, requests, status);
        #ifdef SHOW_TIME
            TIMER_RESET();
        #endif
        handle_reduce(datatype, op, &fma_recv_ops, recv_buffer, data, dst, ft_ctx);
        #ifdef SHOW_TIME
            TIMER_RECORD(time_reduce);
        #endif
        MPI_Barrier(comm);
        block_send = (block_send == 0 ? ft_ctx.num_nodes - 1 : block_send - 1);
        block_recv = (block_recv == 0 ? ft_ctx.num_nodes - 1 : block_recv - 1);
    }
    //LOG_IF(WARNING, node_label == 0) << "gathering done";
    for (size_t i = 0; i != ft_ctx.num_nodes - 1; i++)
    {
        std::vector<FMA_Operation> fma_send_ops = {FMA_Operation(right, block_send, ft_ctx.num_nodes, ft_ctx.data_size, false)};
        std::vector<FMA_Operation> fma_recv_ops = {FMA_Operation(left, block_recv, ft_ctx.num_nodes, ft_ctx.data_size, false)};
        request_index = handle_send(comm, datatype, &fma_send_ops, dst, ft_ctx, requests);
        request_index += handle_recv(comm, datatype, &fma_recv_ops, dst, ft_ctx, requests + request_index);
        MPI_Waitall(request_index, requests, status); 
        MPI_Barrier(comm);
        block_send = (block_send == 0 ? ft_ctx.num_nodes - 1 : block_send - 1);
        block_recv = (block_recv == 0 ? ft_ctx.num_nodes - 1 : block_recv - 1);
    }
    //LOG_IF(WARNING, node_label == 0) << "broadcast done";
    delete[] requests;
    delete[] status;
}

} // end of namespace FlexTree

#ifdef STANDALONE_TEST
int MPI_Allreduce_FT(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
#else
static int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
#endif
{
    #ifdef FT_DEBUG
    std::cout << "FlexTree AR called" << std::endl;
    #endif
    std::pair<std::vector<size_t>, size_t> topo = FlexTree::get_stages(FlexTree::FlexTree_Context::get_num_nodes(comm));
    auto stages = topo.first;
    const FlexTree::FlexTree_Context ft_ctx(comm, datatype, count, topo.second);
    #ifdef FT_DEBUG
    if (ft_ctx.node_label == 0) ft_ctx.show_context();
    #endif

    if (ft_ctx.num_nodes <= 1)
    {
        if (sendbuf != MPI_IN_PLACE)
        {
            memcpy(recvbuf, sendbuf, count * ft_ctx.type_size);
        }
        return 0;
    }

    FlexTree::recv_buffer = FlexTree::flextree_register_the_buffer((ft_ctx.data_size_aligned * ft_ctx.type_size) << 1);
    
    // MPI_IN_PLACE
    if (stages[0] != 1)
    {
        if (sendbuf == MPI_IN_PLACE)
        {
            FlexTree::tree_allreduce(datatype, op, comm, nullptr, recvbuf, ft_ctx, stages);
        }
        else 
        {
            FlexTree::tree_allreduce(datatype, op, comm, sendbuf, recvbuf, ft_ctx, stages);
        }
    }
    else
    {
        if (sendbuf == MPI_IN_PLACE)
        {
            FlexTree::ring_allreduce(datatype, op, comm, nullptr, recvbuf, ft_ctx);
        }
        else 
        {
            FlexTree::ring_allreduce(datatype, op, comm, sendbuf, recvbuf, ft_ctx);
        }
    }
    
#ifdef FT_DEBUG
    std::cout << "node " << ft_ctx.node_label << " FlexTree AR finished" << std::endl;
#endif
    return 0;
}

#endif //end if of check c++
#endif
//end of flextree mod