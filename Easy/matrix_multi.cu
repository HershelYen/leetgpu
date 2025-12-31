#include <cuda_runtime.h>
#define TILE 16 
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
        // 获取x y
        __shared__ float matrix_a[TILE][TILE];
        __shared__ float matrix_b[TILE][TILE];

        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        
        float value = 0.0f;

        for(int ph=0;ph<(N + TILE - 1)/TILE; ++ph){
            // 搬运a
            int a_row = row;
            int a_col = ph*TILE+threadIdx.x;
            
            if( a_row<M && a_col<N){
                matrix_a[threadIdx.y][threadIdx.x] = A[a_row * N+a_col];
            }else{
                matrix_a[threadIdx.y][threadIdx.x]= 0.0f;
            }

            // 搬运b
            int b_row = ph*TILE + threadIdx.y;
            int b_col = col;
            if(b_col<K && b_row<N){
                matrix_b[threadIdx.y][threadIdx.x] = B[b_row * K + b_col];
            }else{
                matrix_b[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // 同步数据
            __syncthreads();

            for(int k=0;k<TILE;k++){
                value+= matrix_a[threadIdx.y][k] * matrix_b[k][threadIdx.x];
            }
            __syncthreads();

        } 
        // 写回数据
        if(row<M && col<K)  C[row*K+col] = value;

    }

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
