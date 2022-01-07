#include<vector>
#include<iostream>
#include<assert.h>

class Candidates
{
private:
    size_t num_nodes;
    std::vector<size_t> tree_now;
    size_t incast_thresh = 9;
    bool check_valid()const 
    {
        assert(!tree_now.empty());
        if (tree_now[0] == 0) return true;
        if (tree_now.size() < 3) return false; // h >= 2
        if (tree_now[0] >= tree_now[1] || tree_now[0] * (tree_now[1] + 1) >= num_nodes) return false; // n < min(w_0, m / w_0)
        for (size_t i = tree_now.size() - 2; i >= 2; --i)
        {
            if (tree_now[0] <= tree_now[i]) return true;
            // n <= max(w_i) where i = 1...h-2
        }
        if (tree_now.size() == 3 && tree_now[2] >= tree_now[0]) return true;
        return false;
    }
    void _generate(size_t now)
    {
        assert(now != 0);
        if (now == 1)
        {
            if (check_valid()) candidates.push_back(tree_now);
            /*
            else 
            {
                std::cout << "not valid: ";
                for (auto i:tree_now) std::cout << i << " ";
                std::cout << std::endl;
            }
            */
            return;
        }

        /**
         * @brief 高级剪枝
         * 如果剩下的节点不触发 incast, 那直接 full-mesh
         */
        if (now <= incast_thresh)
        {
            tree_now.push_back(now);
            _generate(1);
            tree_now.pop_back();
        }
        else 
        {
            size_t trial_start = 2, trial_end = now;

            /**
             * @brief 高级剪枝
             * insight: 从小于等于 incast_thresh 的第一个因子开始往上遍历. 
             */
            if (incast_thresh > 1)
            {
                trial_start = incast_thresh;
                while(now % trial_start != 0) --trial_start;
                if (trial_start == 1) trial_start = incast_thresh;
            }
            
            /** 
             * @brief 平凡剪枝
             * 优化 w_0 的选择. 因为要求 w_0 > n. 
             * 此处直接将 start 抬到了 n + 1, 是因为一定 n + 1 <= now. 
             */
            if (tree_now.size() == 1 && trial_start < tree_now[0] + 1) trial_start = tree_now[0] + 1;

            /**
             * @brief 平凡剪枝
             * 优化 w_0 的选择, 因为要求 m / w_0 > n. 一旦不满足这个要求, 因为 w_0 递增, 所以后面的也不可能满足. 
             * 式子经过了一定的变形. 
             */
            if (tree_now.size() == 1 && tree_now[0] != 0) trial_end = (num_nodes/tree_now[0]) - 1;

            for (size_t i = trial_start; i <= trial_end; ++i)
            {
                /**
                 * @brief 高级剪枝
                 * insight: 不产生 incast 的 w_i 子区间内,  w_i (非严格) 递减一定是最好的. 
                 */
                //if (tree_now.size() > 1 && *(tree_now.end() - 1) <= incast_thresh && i > *(tree_now.end() - 1) && i <= incast_thresh) continue;
                if (now % i == 0)
                {
                    tree_now.push_back(i);
                    _generate(now / i);
                    tree_now.pop_back();
                }
            }
        }
        
        
    }
public:
    std::vector< std::vector<size_t> > candidates;
    Candidates(size_t n, size_t _incast_thresh = 0x3f3f3f3f): num_nodes(n), incast_thresh(_incast_thresh)
    {
        generate();
    }
    void generate()
    {
        if (tree_now.empty() && candidates.empty())
        {
            /**
             * @brief 剪枝. 
             * 公式推导可以得出下面这个限制. 
             */
            for (size_t n = 0; n * (n + 1) < num_nodes; ++n)
            {
                tree_now.push_back(n);
                _generate(num_nodes - n);
                tree_now.clear();
            }
            
        }
    }
    void print()const 
    {
        for (const auto &i:candidates)
        {
            for (const auto &j:i)
            {
                std::cout << j << " ";
            }
            std::cout << std::endl;
        }
    }
};
