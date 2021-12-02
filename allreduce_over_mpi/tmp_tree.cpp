// g++ -o build/tmp_tree tmp_tree.cpp && ./build/tmp_tree

#include<iostream>
#include<vector>
#include<assert.h>
#include<string>
#include<sstream>

namespace FlexTree
{

const int INF = 0x3F3F3F3F;

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
     * @brief 看给的这个节点下面是否覆盖了 lonely blocks 的区间.
     * @param node_label (default: this->node_label)
     * @return bool true/false
     */
    inline bool has_lonely_blocks(size_t n = INF) const 
    {
        if (n == INF) n = node_label;
        assert(n < num_split);
        return (num_lonely > 0) && (n >= stages[0] * (num_lonely)) && (n % stages[0] < num_lonely);
        /**
         * 值得注意的三个条件:
         * 1. 首先, 全局得存在孤立节点, 才有这个问题.
         * 2. 其次, 该节点应当在孤立区内, 即子节点所覆盖的范围和孤立区重合.
         * 3. 最后, 这个节点本身在最底层得负责一块孤立.
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
        if (has_lonely_blocks(n)) return ans;
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
                    if (has_lonely_blocks())
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
            // 最后一步不发东西
            lonely_ops.emplace_back();
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
                    if (followers.size() > 0 && has_lonely_blocks(left_peer) && i != stages.size() - 1)
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
            // 最后一步要从不孤独的人们那儿收点税
            {
                std::vector<Operation> stage_lonely_ops;
                size_t gap = num_split / *(stages.end() - 1);
                for (int i = node_label - stages[0]; i >= 0; i -= gap)
                {
                    if (find_followers(stages.size() - 1, i).size() == 1)
                    {
                        assert(find_followers(stages.size() - 1, i)[0] == node_label);
                        stage_lonely_ops.emplace_back(i, node_label);
                    }
                }
                lonely_ops.push_back(stage_lonely_ops);
            }
        }
    }
};

/**
 * FMA = FlexTree-MPI Adapter
 * 用这种缩写看起来就很高大上
 * 其实很简单: 就是标明向谁发哪块内存地址的数据
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
     * 
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

} //end of namespace

int main()
{
    using namespace std;
    using namespace FlexTree;
    cout << "----- Test of tree generator -----" << endl;
for (size_t i = 20; i != 25; i++){
    Send_Operations send(26, 2, i, {4,3,2});
    Recv_Operations recv(26, 2, i, {4,3,2});
    FMA_Send_Operations fma_send(&send, &recv, 26, 590);
    FMA_Recv_Operations fma_recv(&send, &recv, 26, 590);

    send.generate(); recv.generate();
    fma_send.generate();
    fma_recv.generate();
    send.print();
    cout << " --- " << endl;
    fma_send.print();
    cout << " --- " << endl;
    recv.print();
    cout << " --- " << endl;
    fma_recv.print();
    cout << endl << endl;
}
    return 0;
}