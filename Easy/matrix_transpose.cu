#include <cuda_runtime.h>
#define TILE 16
#define CEIL(M, N) (M+N-1)/N

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // 定义shared memory
    __shared__ float temp[TILE][TILE+1];
    // 获取Idx
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    // row_id
    int row_id = blockIdx.y * blockDim.y + threadIdx.y;
    int col_id = blockIdx.x * blockDim.x + threadIdx.x;

    // i->j j-> i

    // 写tile,16 * 16 线程,一个线程写一个数
    if(row_id<rows && col_id<cols){
        temp[tidy][tidx] = input[row_id * cols + col_id];
    } else{
        temp[tidy][tidx] = 0.0f;
    }
    int x_out = blockIdx.y * TILE + tidx;
    int y_out = blockIdx.x * TILE + tidy;

    __syncthreads();
    // 确定转置的row与col
    // 转置写回
    if(x_out<rows && y_out<cols){
        output[y_out*rows + x_out] = temp[tidx][tidy];
    }
    
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid(CEIL(cols, threadsPerBlock.x), CEIL(rows, threadsPerBlock.y));

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
