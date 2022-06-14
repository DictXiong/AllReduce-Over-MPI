/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <iostream>
#include <vector>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "time_record.h"
#include "reduce_sum.h"
#include "reduce_sum_gpu.h"

#define cuda_error_check(err,msg) do {LOG_IF(FATAL, err != cudaSuccess) << msg << ", err: " << cudaGetErrorString(err);} while (false)

template<typename T> __host__
void print_vector(const std::vector<T> vec)
{
    for (const auto &i : vec)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements num_elements.
 */

__host__
std::pair<size_t, size_t> benchmark(const uint num_blocks, const uint num_elements)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    const size_t size = num_elements * sizeof(float);
    std::cout << "----------" << std::endl;
    LOG(WARNING) << "Vector addition of " << num_blocks << " * " << num_elements << " elements";

    // Allocate memory on the host
    float **h_vectors = new float *[num_blocks];
    for (uint i = 0; i != num_blocks; ++i)
    {
        h_vectors[i] = (float *)malloc(size);
        LOG_IF(FATAL, h_vectors[i] == NULL) << "Failed to allocate host vectors";
    }
    float *h_result = (float *)malloc(size);
    float *h_result_from_d = (float *)malloc(size);
    // Verify that allocations succeeded
    LOG_IF(FATAL, h_result == NULL || h_result_from_d == NULL) << "Failed to allocate host vectors";

    // Initialize the host input vectors
    for (uint i = 0; i < num_elements; ++i)
    {
        for (uint j = 0; j < num_blocks; ++j)
        {
            h_vectors[j][i] = rand()/(float)RAND_MAX;
        }
    }

    // Allocate memory on the device
    float **d_vectors = new float *[num_blocks];
    for (uint i = 0; i != num_blocks; ++i)
    {
        err = cudaMalloc((void **)&d_vectors[i], size);
        cuda_error_check(err, "Failed to allocate device vectors");
    }
    float *d_result;
    err = cudaMalloc((void **)&d_result, size);
    //if (err != cudaSuccess) LOG(INFO) << "ye? " << cudaGetErrorString(err);
    cuda_error_check(err, "Failed to allocate device vector d_result");

    // Copy the host input vectors in host memory to the device input vectors in
    // device memory
    LOG(INFO) << "Copy input data from the host memory to the CUDA device";
    for (uint i = 0; i != num_blocks; ++i)
    {
        err = cudaMemcpy(d_vectors[i], h_vectors[i], size, cudaMemcpyHostToDevice);
        cuda_error_check(err, "Failed to copy host vectors to device vectors");
    }

    // Launch the Vector Add CUDA Kernel
    uint threadsPerBlock = 256;
    uint blocksPerGrid =(num_elements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    std::pair<uint, uint> ans; 
    // GPU
    {
        newplan::Timer timer;
        cudaDeviceSynchronize();
        timer.Start();
        //vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);
        reduce_sum_gpu((const float**)d_vectors, (float*)d_result, num_blocks, num_elements, blocksPerGrid, threadsPerBlock);
        cudaDeviceSynchronize();
        timer.Stop();
        ans.first = timer.MilliSeconds();
        //std::cout << "GPU Time: " << timer.MilliSeconds() << "ms" << std::endl;
    }
    
    err = cudaGetLastError();
    cuda_error_check(err, "Failed to launch vectorAdd kernel");

    // CPU 
    {
        newplan::Timer timer;
        timer.Start();
        reduce_sum((const float**)h_vectors, (float*)h_result, num_blocks, num_elements);
        timer.Stop();
        ans.second = timer.MilliSeconds();
        //std::cout << "CPU Time: " << timer.MilliSeconds() << "ms" << std::endl;
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    LOG(INFO) << "Copy output data from the CUDA device to the host memory";
    err = cudaMemcpy(h_result_from_d, d_result, size, cudaMemcpyDeviceToHost);
    cuda_error_check(err, "Failed to copy vectors from device to host");

    // Verify that the result vector is correct
    for (uint i = 0; i < num_elements; ++i)
    {
        if (fabs(h_result[i] - h_result_from_d[i]) > 1e-5)
        {
            LOG(FATAL) << "Result verification failed at element " << i << std::endl << "CPU: " << h_result[i] << ", GPU: " << h_result_from_d[i];
            // fprintf(stderr, "Result verification failed at element %d!\n", i);
            // fprintf(stderr, "CPU: %f, GPU: %f\n", h_C[i], hd_C[i]);
        }
    }

    LOG(WARNING) << "Test PASSED";

    // Free device global memory
    for (uint i = 0; i != num_blocks; ++i)
    {
        err = cudaFree(d_vectors[i]);
        cuda_error_check(err, "Failed to free device vector");
    }
    err = cudaFree(d_result);
    cuda_error_check(err, "Failed to free device vector");

    // Free host memory
    for (uint i = 0; i != num_blocks; ++i)
    {
        free(h_vectors[i]);
    }
    free(h_result);
    free(h_result_from_d);
    LOG(INFO) << "Time in ms, GPU: " << ans.first << ", CPU: " << ans.second;
    return ans;
}

/**
 * Host main routine
 */
int main(int argc, char **argv)
{
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();


    // Print the vector length to be used, and compute its size
    std::vector<uint> results_cpu, results_gpu;
    for (uint i = 1; i != 9; ++i)
    {
        auto ret = benchmark(i, 400e6);
        results_gpu.push_back(ret.first);
        results_cpu.push_back(ret.second);
    }
    print_vector(results_gpu);
    print_vector(results_cpu);

    return 0;
}

