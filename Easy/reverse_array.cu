#include <cuda_runtime.h>

// simple one, you just need half of the threads to swap the array
__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N / 2){
        float tmp = input[idx];
        // 计算需要反转的idx
        input[idx] = input[N-idx-1];
        input[N-idx-1] = tmp;
    }
}



// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
