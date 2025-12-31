#

## Vector add
A B 对应位置相加，然后写入C对应位置

## Matrix multiplication

### 基础版
采用分tile的方式进行计算，可以画图理解一下，计算Tile x Tile大小的 C 矩阵，需要A的Tile 行 和 B 的 Tile 列。同时对A和B的N维度也进行分块处理，想清楚C对应位置需要A和B哪些数据。要记得边界补0

> 关于TILE和BLOCK的选取  
TILE和BLOCK的大小选取是一个性能调优(Tuning)和硬件限制(Hardware Limits)的核心问题
- 对于Tile size
    1. 受到单个block最大 1024 线程限制。单个block总线程数为TILE.X x TILE.Y，最好不要选择满线程1024。
    2. shared memory大小限制  16✖️16✖️4 Byte = 2KB，这对于Shared memory 64KB+ 的GPU毫无压力
- 对于Block size
    1. 保证block足够覆盖C的每一行与每一列(Total + threadsPerBlock -1)/threadsPerBlock是标准的向上取整







## Reverse Array
### 基础版
Reverse Array是一道相对比较简单的题目，我提供一个相对比较简单的解法
```cuda
    if(idx<N / 2){
        float tmp = input[idx];
        // 计算需要反转的idx
        input[idx] = input[N-idx-1];
        input[N-idx-1] = tmp;
    }
```
这个解法是最简单的解法，但这个解法的问题在于当我们读取idx的时候是正序读取，这很好。但是N-idx-1是倒序读取的，这会带来性能的一个巨大隐患：非合并访问(Uncoalesced Access)

### 进阶版
当涉及到非合并访问，一般可以利用share memory 作为中转站，将“倒序读写”转变为“正序读写”

核心思想
1. 分块：每个block处理数组的左边一块和对应右边一块
2. 正序读取左右两块
3. 内部反转
4. 正序写回

