#include<iostream>
#include<sstream>
#include<fstream>
#include<vector>
#include<string.h>
#include<thread>
#include<stdlib.h>
#include<cmath>
#include<mpi.h>
#include<glog/logging.h>
#define STANDALONE_TEST
#include "mpi_mod.hpp"

void show_version()
{
    LOG(WARNING) << std::endl << "----------" << std::endl << "FlexTree Standalone Benchmark" << std::endl << "version: " << GIT_REPO_VERSION << std::endl << "date: " << GIT_REPO_DATE << std::endl << "hash: " << GIT_REPO_HASH << std::endl;
}

// util
template<typename T>
void write_vectors_to_file(std::string filename, std::initializer_list<std::vector<T>> vectors)
{
    std::ofstream f(filename, std::ios::out);
    size_t len = vectors.begin()->size();
    for (size_t i = 0; i < len; ++i)
    {
        for (const auto &j : vectors)
        {
            f << j[i] << "\t";
        }
        f << std::endl;
    }
    f.close();
}

#ifdef SHOW_TIME
extern double time_reduce;
#endif

int main(int argc, char **argv)
{
        // 当前节点的编号, 总结点数量, 孤立节点数量
    size_t node_label, total_peers, num_lonely = 0; 

    // 命令行参数
    int repeat = 1;
    int comm_type = 0; // 0 for tree, 1 for ring, 2 for mpi
    bool to_file = false;
    bool check_validity = false;
    size_t data_len = 35;
    std::string tag;

    // others
    std::vector<double> repeat_time;
    double sum_time = 0, min_time = FlexTree::INF;
    #ifdef SHOW_TIME
    std::vector<double> repeat_time_reduce;
    double sum_time_reduce = 0;
    #endif
    int tmp;

    // init mpi
    MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE, &tmp);
    MPI_Comm_size(MPI_COMM_WORLD, &tmp);
    total_peers = tmp;
    MPI_Comm_rank(MPI_COMM_WORLD, &tmp);
    node_label = tmp;
    // end init

    auto topo = FlexTree::get_stages(total_peers);

    // init glog
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    //if (node_label == 0) 
        google::InstallFailureSignalHandler();
    // end init
    LOG(INFO) << "Hi here's " << node_label;
    
    // arg parse
    for (auto i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--size") == 0)
        {
            i++;
            CHECK_GE(argc, i);
            std::istringstream ss(argv[i]);
            ss >> data_len;
        }
        else if (strcmp(argv[i], "--repeat") == 0)
        {
            i++;
            CHECK_GE(argc, i);
            std::istringstream ss(argv[i]);
            ss >> repeat;
        }
        else if (strcmp(argv[i], "--to-file") == 0)
        {
            to_file = true;
        }
        else if (strcmp(argv[i], "--comm-type") == 0)
        {
            i++;
            CHECK_GE(argc, i);
            if (strcmp(argv[i], "mpi") == 0)
            {
                comm_type = 2;
            }
            else if (strcmp(argv[i], "flextree") == 0)
            {
                comm_type = 0;
            }
        }
        else if (strcmp(argv[i], "--tag") == 0)
        {
            i++;
            CHECK_GE(argc, i);
            std::istringstream ss(argv[i]);
            ss >> tag;
        }
        else if (strcmp(argv[i], "--version") == 0)
        {
            show_version();
            exit(0);
        }
        else if (strcmp(argv[i], "--check") == 0)
        {
            check_validity = true;
        }
        else
        {
            LOG(FATAL) << "unknown parameter: " << argv[i];
        }
    }

    // 初始化 data 和 buffer
    float *data = new float[data_len];
    const float init_data_base = 0.1;
    for (size_t i = 0; i != data_len; i++)
    {
        data[i] = i * init_data_base;
    }
    auto recvbuf =(void*)(new float[data_len]);
    // 各种初始化完成
    MPI_Barrier(MPI_COMM_WORLD);
    // 打印设置总结
    if (node_label == 0)
    {
        std::ostringstream ss;
        ss << "configuration: \n  - total_peers: "<< total_peers << "\n  - data_size: " << data_len << "\n  - repeat: " << repeat << "\n  - to_file: " << (to_file ? "true":"false") << "\n  - check_validity: " << (check_validity ? "true" : "false");
        if (to_file && !tag.empty()) ss << "\n  - file tag: " << tag;
        ss << "\n  - communication method: " << (comm_type ? "mpi":"flextree");
        if (comm_type == 0)
        {
            ss << "\n  - And FlexTree topo is ";
            for (auto i:topo.first)
            {
                ss << i << " ";
            }
            ss << "+" << topo.second << " ";
        }
        LOG(WARNING) << "\n" << ss.str();
    }
    LOG_IF(INFO, node_label == 0) << "sleep for 2 senconds";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    // 准备就绪
    if (comm_type == 0) //tree
    {
        for (auto i = 0; i < repeat; i++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            #ifdef SHOW_TIME
            time_reduce = 0;
            #endif
            auto time1 = MPI_Wtime();
            MPI_Allreduce_FT(MPI_IN_PLACE, data, data_len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            auto time2 = MPI_Wtime();
            #ifdef SHOW_TIME
            repeat_time_reduce.push_back(time_reduce);
            sum_time_reduce += time_reduce;
            #endif
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
            #ifdef SHOW_TIME
            time_reduce = 0;
            #endif
            auto time1 = MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE, data, data_len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            auto time2 = MPI_Wtime();
            #ifdef SHOW_TIME
            repeat_time_reduce.push_back(time_reduce);
            sum_time_reduce += time_reduce;
            #endif
            repeat_time.push_back(time2 - time1);
            sum_time += time2 - time1;
            min_time = std::min(time2 - time1, min_time);
            LOG_IF(WARNING, node_label == 0) << "repeat " << i << " finished"; 
        }
    }
    else 
    {
        LOG(FATAL) << "unknown comm type: " << comm_type;
    }

    for (int i = 0; i <= total_peers; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == node_label + 1)
        {
            std::cout << "CHECK " << node_label << ": ";
            for (int i = 9; i != 24; i++) std::cout << data[i] << " ";
            if (check_validity) // 如果打算进行正确性校验
            {
                bool valid = true;
                size_t invalid_pos = 0;
                for (int i = 0; i < data_len; i++)
                {
                    if (data[i] - i * init_data_base * pow(total_peers, repeat) > init_data_base * 0.1)
                    {
                        valid = false;
                        invalid_pos = i;
                        break;
                    }
                }
                if (valid) std::cout << "(test passed)";
                else LOG(WARNING) << "node " << node_label << " says the result seems not right at " << invalid_pos << ". Maybe the result is too large?";
            }
            std::cout << std::endl;
        }
    }
    
    MPI_Finalize();

    // 写入文件
    if (node_label == 0 && to_file)
    {
        std::ostringstream ss;
        if (!tag.empty()) ss << tag << ".";
        ss << total_peers << "." << data_len << ".";
        if (comm_type == 0)
        {
            for (auto i : topo.first)
            {
                ss << i << "-";
            }
            ss << "+" << topo.second;
        }
        else
        {
            ss << "mpi";
        }
        ss << (FlexTree::comm_only ? ".comm_test." : ".ar_test.");
        ss << time(NULL) << ".txt";
        #ifdef SHOW_TIME
        write_vectors_to_file(ss.str(), {repeat_time, repeat_time_reduce});
        #else
        write_vectors_to_file(ss.str(), {repeat_time});
        #endif
    }

    LOG_IF(WARNING, node_label == 0) << "\nDONE, average time: " << sum_time / repeat << ", min time: " << min_time << std::endl;
    #ifdef SHOW_TIME
    LOG_IF(INFO, node_label == 0) << "And average reduce time: " << sum_time_reduce / repeat << std::endl;
    #endif
    google::ShutdownGoogleLogging();

    return 0;
}