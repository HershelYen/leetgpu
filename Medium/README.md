## Reduction
### 基础版（树状规约）

自定义threads和blocks大小

1. 首先对N进行规约，将N规约到block大小
2. 然后block内部规约

> 这种方法的缺点在于没有充分发挥SIMT的优势

### 进阶版（线程粗化 + 树状规约 + Warp内规约）

这一部分参考[reduction_adv](./reduction.adv.cu)

说一说可能要注意的细节或遇到的bug

1. volatile修饰，在warp内规约时，一定要注意volatile修饰，不然会被编译器优化读写导致结果错误。
```C++
__device__ void warp_reduce(volatile float *sdata, int tid){
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
}
```


2. 宏定义问题。编译器处理宏定义是原位替换，这导致我的结果虽然是正确的，但是我的速度非常慢，最后检查代码发现在启动kernel的时候经过宏定义替换后多启动了将近64倍的block数量，引发了严重的原子操作冲突和调度开销。
```C++
// 此处定义NUM_BLOCK
#define NUM_BLOCK THREADS*STRIDE
// 在替换后
int blockPerGrid = DIV_CEIL(N, NUM_BLOCK);
// 变成了以下内容
#define DIV_CEIL(a, b) (N + THREADS*STRIDE - 1)/ THREADS * STRIDE
```

所以说尽量少用宏定义，替换成以下代码
```C++
const int threads_per_block = 256;
const int stride = 8;
const int num_element_per_block = threads_per_block * stride;
```