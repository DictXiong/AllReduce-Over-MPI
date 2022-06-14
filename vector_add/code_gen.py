def gen_reduce_sum_gpu_subfuncs():
    for i in range(1,21):
        print(f"template<class DataType>\n__global__\nvoid reduce_sum_{i}(", end='')
        for j in range(i):
            print(f"const DataType *src{j}, ", end='')
        print("DataType *dst, const size_t num_elements)\n{")
        print("    int i = blockDim.x * blockIdx.x + threadIdx.x;")
        print("    if (i < num_elements)")
        print("    {")
        print("        dst[i] = ", end="")
        for j in range(i - 1):
            print(f"src{j}[i] + ", end="")
        print(f"src{i-1}[i];")
        print("    }")
        print("}")

def gen_reduce_sum_gpu():
    gen_reduce_sum_gpu_subfuncs()
    print()
    print("""template<class DataType>
__host__ void reduce_sum_gpu(const DataType **src, DataType *dst, const int num_blocks, const size_t num_elements, int blocksPerGrid, int threadsPerBlock)
{
    if (num_blocks <= 1) return;
    switch (num_blocks)
    {""")
    for i in range(1,21):
        print(f"    case {i}:")
        print("    {")
        print(f"        reduce_sum_{i}<DataType><<<blocksPerGrid, threadsPerBlock>>>(", end='')
        for j in range(i):
            print(f"src[{j}], ", end='')
        print("dst, num_elements);")
        print("        break;")
        print("    }")
    print("    }\n}")




gen_reduce_sum_gpu()