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
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "time_record.h"
#include "reduce_sum.h"
#include "reduce_sum_gpu.h"
#include <glog/logging.h>

#define cuda_error_check(err,msg) do {LOG_IF(FATAL, err != cudaSuccess) << msg << ", err: " << cudaGetErrorString(err);} while (false)

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements num_elements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        C[i] = A[i] + B[i];
    }
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
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int num_elements = 500e6;
    const int num_blocks = 4;
    const size_t size = num_elements * sizeof(float);
    LOG(WARNING) << "[Vector addition of " << num_blocks << " * " << num_elements << " elements]";

    // Allocate memory on the host
    float **h_vectors = new float *[num_blocks];
    for (int i = 0; i != num_blocks; ++i)
    {
        h_vectors[i] = (float *)malloc(size);
        LOG_IF(FATAL, h_vectors[i] == NULL) << "Failed to allocate host vectors";
    }
    float *h_result = (float *)malloc(size);
    float *h_result_from_d = (float *)malloc(size);
    // Verify that allocations succeeded
    LOG_IF(FATAL, h_result == NULL || h_result_from_d == NULL) << "Failed to allocate host vectors";

    // Initialize the host input vectors
    for (int i = 0; i < num_elements; ++i)
    {
        for (int j = 0; j < num_blocks; ++j)
        {
            h_vectors[j][i] = rand()/(float)RAND_MAX;
        }
    }

    // Allocate memory on the device
    float **d_vectors = new float *[num_blocks];
    for (int i = 0; i != num_blocks; ++i)
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
    for (int i = 0; i != num_blocks; ++i)
    {
        err = cudaMemcpy(d_vectors[i], h_vectors[i], size, cudaMemcpyHostToDevice);
        cuda_error_check(err, "Failed to copy host vectors to device vectors");
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(num_elements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // GPU
    {
        newplan::Timer timer;

        timer.Start();
        //vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);
        reduce_sum_gpu((const float**)d_vectors, (float*)d_result, num_blocks, num_elements, blocksPerGrid, threadsPerBlock);
        cudaDeviceSynchronize();
        timer.Stop();
        std::cout << "GPU Time: " << timer.MilliSeconds() << "ms" << std::endl;
    }
    
    err = cudaGetLastError();
    cuda_error_check(err, "Failed to launch vectorAdd kernel");

    // CPU 
    {
        newplan::Timer timer;
        timer.Start();
        reduce_sum((const float**)h_vectors, (float*)h_result, num_blocks, num_elements);
        timer.Stop();
        std::cout << "CPU Time: " << timer.MilliSeconds() << "ms" << std::endl;
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    LOG(INFO) << "Copy output data from the CUDA device to the host memory";
    err = cudaMemcpy(h_result_from_d, d_result, size, cudaMemcpyDeviceToHost);
    cuda_error_check(err, "Failed to copy vectors from device to host");

    // Verify that the result vector is correct
    for (int i = 0; i < num_elements; ++i)
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
    for (int i = 0; i != num_blocks; ++i)
    {
        err = cudaFree(d_vectors[i]);
        cuda_error_check(err, "Failed to free device vector");
    }
    err = cudaFree(d_result);
    cuda_error_check(err, "Failed to free device vector");

    // Free host memory
    for (int i = 0; i != num_blocks; ++i)
    {
        free(h_vectors[i]);
    }
    free(h_result);
    free(h_result_from_d);
    LOG(INFO) << "Done";
    return 0;
}

