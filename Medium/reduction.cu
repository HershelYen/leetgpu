#include <cuda_runtime.h>
// 定义256 ts per block
#define THREADS 256
// 定义256 block per grid
#define BLOCKS 256

__global__ void reduction_kernel(const float* input, float* output, int N){
    // 定义shared mem(单个block内共享)
    __shared__ float sdata[THREADS];

    // 定义线程内局部变量
    float local_sum = 0.0f;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 第一步，将超出grid线程总数的范围进行规约(s[0] + s[0+stride] + s[2*stride])
    for(int i=idx; i<N; i+=stride){
        local_sum+=input[i];
    }
    // 同步
    // 每个线程都有一个local sum，将local_sum存到共享的sdata
    sdata[tid] = local_sum;
    __syncthreads();

    // 第二步，单个block内部进行规约
    for(int s = blockDim.x/2; s>0; s>>=1){
        if(tid<s){
            sdata[tid] +=sdata[tid+s];
        }
        __syncthreads();
    }

    
    // 循环结束后，sdata[0]存储了各个block结果，汇总
    if(tid==0){
        atomicAdd(output, sdata[0]);
    }

}
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = THREADS;
    int blocksPerGrid = BLOCKS;
    // 启动kernel
    reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
