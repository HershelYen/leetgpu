#include <cuda_runtime.h>
#define THREADS 256
#define STRIDE 8
#define WARP_SIZE 32
#define NUM_BLOCK (THREADS*STRIDE)
#define ceil(a, b) ((a)+(b)-1)/(b)
__device__ void warp_reduce(volatile float *sdata, int tid){
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
}

__global__ void reduction_kernel(const float* input, float* output, int N){
    __shared__ float sdata[THREADS];
    int tid = threadIdx.x;
    // 找到block起始位置,block id 和每个block处理的数据数量
    int block_start = blockIdx.x * NUM_BLOCK;
    // 第一步，256*8 -> 256
    float sum = (block_start + tid) < N?input[block_start+tid]:0.0f;
    for(int i=1; i<STRIDE;++i){
        // 定义需要规约的id
        int gid =  block_start + i * THREADS + tid;
        if(gid<N){
            sum += input[gid];
        }
    }
    // sum放到sdata中
    sdata[tid] = sum;
    __syncthreads();

    // -----------------------------------------------------
    // 阶段 2：Block 内树状归约 (Shared Memory)
    // 从 128 -> 64 -> 32
    // -----------------------------------------------------
    for(int s = blockDim.x/2; s>WARP_SIZE; s>>=1){
        if(tid<s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    // -----------------------------------------------------
    // 阶段 3：Warp 内归约 (32 -> 1)
    // -----------------------------------------------------
    if(tid<WARP_SIZE){
        warp_reduce(sdata, tid);
    }
    if(tid==0){
        atomicAdd(output, sdata[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = THREADS;
    int blockPerGrid = ceil(N, NUM_BLOCK);
    // cudaMemset(output, 0, sizeof(float));
    // 启动kernel
    reduction_kernel<<<blockPerGrid, threadsPerBlock>>>(input, output, N);
}
