#include <cuda_runtime.h>
#define BLOCK_SIZE 512
// idx 和 N-idx
__global__ void reverse_array(float* input, int N) {
    __shared__ float s_left[BLOCK_SIZE];
    __shared__ float s_right[BLOCK_SIZE];

    int left_block_start = blockIdx.x * blockDim.x;
    // 左边结束位置 (bid+1)dim
    // 右边开始位置 N-(bid+1)dim = N - left - dim
    int right_block_start = N - (left_block_start + blockDim.x);

    // 线程索引
    int tid = threadIdx.x;
    int left_idx = left_block_start + tid;
    int right_idx = right_block_start + tid;

    float l_val = 0.0f;
    float r_val = 0.0f;

    if(left_idx<N) l_val = input[left_idx];
    if (right_idx > 0) r_val = input[right_idx];


    // 存入shared_memory
    s_left[tid] = l_val;
    s_right[tid] = r_val;

    __syncthreads();

    int mirror_tid = BLOCK_SIZE - 1 - tid;
    
    if (left_idx < N && left_idx < (N - 1 - left_idx)) {
         input[left_idx] = s_right[mirror_tid];
    }

    // 针对右半部分写入：input[right_idx] 应该变为 s_left[mirror_tid]
    // 只有当 right_idx > (N - 1 - right_idx) 时，说明它是后半部分，需要被写
    if (right_idx > 0 && right_idx < N && right_idx > (N - 1 - right_idx)) {
         input[right_idx] = s_left[mirror_tid];
    }
}



// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = BLOCK_SIZE;
    int elems_per_block = 2 * threadsPerBlock;
    int blocksPerGrid = (N + elems_per_block - 1) / elems_per_block;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
