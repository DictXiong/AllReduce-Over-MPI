#include <cstddef>

template<class DataType>
__global__
void reduce_sum_1(const DataType *src0, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i];
    }
}
template<class DataType>
__global__
void reduce_sum_2(const DataType *src0, const DataType *src1, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i];
    }
}
template<class DataType>
__global__
void reduce_sum_3(const DataType *src0, const DataType *src1, const DataType *src2, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i];
    }
}
template<class DataType>
__global__
void reduce_sum_4(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i];
    }
}
template<class DataType>
__global__
void reduce_sum_5(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i];
    }
}
template<class DataType>
__global__
void reduce_sum_6(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i];
    }
}
template<class DataType>
__global__
void reduce_sum_7(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i];
    }
}
template<class DataType>
__global__
void reduce_sum_8(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i];
    }
}
template<class DataType>
__global__
void reduce_sum_9(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i];
    }
}
template<class DataType>
__global__
void reduce_sum_10(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i];
    }
}
template<class DataType>
__global__
void reduce_sum_11(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, const DataType *src10, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i];
    }
}
template<class DataType>
__global__
void reduce_sum_12(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, const DataType *src10, const DataType *src11, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i];
    }
}
template<class DataType>
__global__
void reduce_sum_13(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, const DataType *src10, const DataType *src11, const DataType *src12, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i];
    }
}
template<class DataType>
__global__
void reduce_sum_14(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, const DataType *src10, const DataType *src11, const DataType *src12, const DataType *src13, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i];
    }
}
template<class DataType>
__global__
void reduce_sum_15(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, const DataType *src10, const DataType *src11, const DataType *src12, const DataType *src13, const DataType *src14, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i];
    }
}
template<class DataType>
__global__
void reduce_sum_16(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, const DataType *src10, const DataType *src11, const DataType *src12, const DataType *src13, const DataType *src14, const DataType *src15, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i] + src15[i];
    }
}
template<class DataType>
__global__
void reduce_sum_17(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, const DataType *src10, const DataType *src11, const DataType *src12, const DataType *src13, const DataType *src14, const DataType *src15, const DataType *src16, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i] + src15[i] + src16[i];
    }
}
template<class DataType>
__global__
void reduce_sum_18(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, const DataType *src10, const DataType *src11, const DataType *src12, const DataType *src13, const DataType *src14, const DataType *src15, const DataType *src16, const DataType *src17, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i] + src15[i] + src16[i] + src17[i];
    }
}
template<class DataType>
__global__
void reduce_sum_19(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, const DataType *src10, const DataType *src11, const DataType *src12, const DataType *src13, const DataType *src14, const DataType *src15, const DataType *src16, const DataType *src17, const DataType *src18, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i] + src15[i] + src16[i] + src17[i] + src18[i];
    }
}
template<class DataType>
__global__
void reduce_sum_20(const DataType *src0, const DataType *src1, const DataType *src2, const DataType *src3, const DataType *src4, const DataType *src5, const DataType *src6, const DataType *src7, const DataType *src8, const DataType *src9, const DataType *src10, const DataType *src11, const DataType *src12, const DataType *src13, const DataType *src14, const DataType *src15, const DataType *src16, const DataType *src17, const DataType *src18, const DataType *src19, DataType *dst, const size_t num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i] + src4[i] + src5[i] + src6[i] + src7[i] + src8[i] + src9[i] + src10[i] + src11[i] + src12[i] + src13[i] + src14[i] + src15[i] + src16[i] + src17[i] + src18[i] + src19[i];
    }
}

template<class DataType>
__host__ void reduce_sum_gpu(const DataType **src, DataType *dst, const int num_blocks, const size_t num_elements, int blocksPerGrid, int threadsPerBlock)
{
    if (num_blocks <= 1) return;
    switch (num_blocks)
    {
case 1:
{
    reduce_sum_1<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], dst, num_elements);
    break;
}
case 2:
{
    reduce_sum_2<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], dst, num_elements);
    break;
}
case 3:
{
    reduce_sum_3<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], dst, num_elements);
    break;
}
case 4:
{
    reduce_sum_4<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], dst, num_elements);
    break;
}
case 5:
{
    reduce_sum_5<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], dst, num_elements);
    break;
}
case 6:
{
    reduce_sum_6<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], dst, num_elements);
    break;
}
case 7:
{
    reduce_sum_7<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], dst, num_elements);
    break;
}
case 8:
{
    reduce_sum_8<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], dst, num_elements);
    break;
}
case 9:
{
    reduce_sum_9<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], dst, num_elements);
    break;
}
case 10:
{
    reduce_sum_10<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], dst, num_elements);
    break;
}
case 11:
{
    reduce_sum_11<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10], dst, num_elements);
    break;
}
case 12:
{
    reduce_sum_12<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10], src[11], dst, num_elements);
    break;
}
case 13:
{
    reduce_sum_13<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10], src[11], src[12], dst, num_elements);
    break;
}
case 14:
{
    reduce_sum_14<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10], src[11], src[12], src[13], dst, num_elements);
    break;
}
case 15:
{
    reduce_sum_15<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10], src[11], src[12], src[13], src[14], dst, num_elements);
    break;
}
case 16:
{
    reduce_sum_16<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10], src[11], src[12], src[13], src[14], src[15], dst, num_elements);
    break;
}
case 17:
{
    reduce_sum_17<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10], src[11], src[12], src[13], src[14], src[15], src[16], dst, num_elements);
    break;
}
case 18:
{
    reduce_sum_18<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10], src[11], src[12], src[13], src[14], src[15], src[16], src[17], dst, num_elements);
    break;
}
case 19:
{
    reduce_sum_19<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10], src[11], src[12], src[13], src[14], src[15], src[16], src[17], src[18], dst, num_elements);
    break;
}
case 20:
{
    reduce_sum_20<DataType><<<blocksPerGrid, threadsPerBlock>>>(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10], src[11], src[12], src[13], src[14], src[15], src[16], src[17], src[18], src[19], dst, num_elements);
    break;
}
    }
}