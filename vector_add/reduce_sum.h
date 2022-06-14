#include <cstddef>
#include <glog/logging.h>

template<class DataType> 
void reduce_sum(const DataType **src, DataType *dst, const int &num_blocks, const size_t &num_elements)
{
#ifdef FT_DEBUG
    //std::cout << "reduce_sum called, ele size = " << sizeof(**src) << ", num_blocks = " << num_blocks << ", num_ele = " << num_elements <<  std::endl;
    //std::cout << "AND: " << src[0][0] << " " << src[1][0] << " " << dst[0] << std::endl;
#endif
    if (num_blocks == 0) return;
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
        LOG(FATAL) << "Unknown num_blocks: " << num_blocks;
    }
}


