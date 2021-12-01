// g++ -o build/tmp_tree tmp_tree.cpp && ./build/tmp_tree

#include<iostream>
#include<vector>
#include<assert.h>

template<class T>
void print_vector(std::vector<T> v)
{
    for (auto i:v)
    {
        std::cout << i << ",";
    }
    std::cout << std::endl;
}

namespace FlexTree
{

const int INF = 0x3F3F3F3F;

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
        std::cout << typeid(*this).name() << " of node " << node_label << " in total " << num_split << " peers: " << std::endl;
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
        if ((node_label < num_split && has_lonely_blocks()) || (node_label >= num_split))
        {
            std::cout << "and " << num_lonely << " lonely node(s):" << std::endl;
            for (const auto &i:lonely_ops)
            {
                if (&i != &*(lonely_ops.end() - 1))
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

} //end of namespace

int main()
{
    using namespace std;
    using namespace FlexTree;
    cout << "----- Test of tree generator -----" << endl;
for (size_t i = 0; i != 27; i++){
    Send_Operations send(27, 3, i, {4,3,2});
    Recv_Operations recv(27, 3, i, {4,3,2});
    //auto watch = send.find_followers(0);
    //print_vector(watch);

    send.generate(); recv.generate();
    send.print();
    cout << " --- " << endl;
    recv.print();
    cout << endl << endl;
}
    return 0;
}