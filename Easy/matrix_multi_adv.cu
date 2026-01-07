// 以下代码为FascinatingChampion1227编写，使用双缓冲技术优化矩阵乘法的CUDA内核实现。
#include <cuda_runtime.h>

#define CEIL(a, b) ((a + b - 1) / b)
#define TM 8           // 每个线程处理的结果矩阵行数
#define TK 8           // 每个线程处理的结果矩阵列数
#define TILE_SIZE 16   // 线程块的基本tile大小
#define BM (TM * TILE_SIZE)  // Block在M维度的大小：8 * 16 = 128
#define BK (TK * TILE_SIZE)  // Block在K维度的大小：8 * 16 = 128

__global__ void matmul_double_buffer(const float* A, const float* B, float* C, int M, int N, int K) {
    // 使用双缓冲技术，为As和Bs各声明两个tile缓冲区
    // As[buffer_id][BM][TILE_SIZE]: 存储A矩阵的tile，大小为128×16
    // Bs[buffer_id][TILE_SIZE][BK]: 存储B矩阵的tile，大小为16×128
    __shared__ float As[2][BM][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][BK];

    // 获取当前线程在block内的坐标
    int tx = threadIdx.x;  // 线程在block内的列索引 (0-15)
    int ty = threadIdx.y;  // 线程在block内的行索引 (0-15)

    // 计算当前block在全局矩阵中的起始位置
    int row_base = blockIdx.y * BM;  // 当前block在矩阵A/C中处理的行起始位置
    int col_base = blockIdx.x * BK;  // 当前block在矩阵B/C中处理的列起始位置

    // 每个线程维护一个TM×TK的小结果矩阵，初始化为0
    // val[8][8]: 每个线程负责计算8行8列的结果
    float val[TM][TK] = {{0.0f}};

    // 计算需要处理多少个tiles（沿着N/K维度）
    // num_tiles: 需要迭代的次数，等于N/TILE_SIZE向上取整
    int num_tiles = CEIL(N, TILE_SIZE);

    // 双缓冲索引管理
    int load_buf_idx = 0;  // 指示当前用于加载数据的缓冲区索引 (0或1)
    int calc_buf_idx = 1;  // 指示当前用于计算的缓冲区索引 (0或1)

    // ==================== 初始数据加载 ====================
    // 在第一次计算之前，先加载第一个tile到缓冲区0
    
    // 加载A矩阵的第一个tile到共享内存As[0]
    // 每个线程负责加载TM行中的特定元素
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        // 计算全局行索引：block起始行 + 线程负责的第i行 + 线程在tile内的行偏移
        int row = row_base + i * TILE_SIZE + ty;
        // 计算在当前tile内的列索引
        int k_idx = tx;
        
        // 边界检查：确保不越界访问
        if (row < M && (0 * TILE_SIZE + k_idx) < N) {
            // 将A矩阵的数据加载到共享内存As的load_buf_idx缓冲区
            // As[buffer][行索引][列索引] = A[全局行][全局列]
            As[load_buf_idx][i * TILE_SIZE + ty][tx] = A[row * N + 0 * TILE_SIZE + k_idx];
        } else {
            // 越界位置填充0
            As[load_buf_idx][i * TILE_SIZE + ty][tx] = 0.0f;
        }
    }

    // 加载B矩阵的第一个tile到共享内存Bs[0]
    // 每个线程负责加载TK列中的特定元素
    #pragma unroll
    for (int i = 0; i < TK; i++) {
        // 计算全局列索引：block起始列 + 线程负责的第i列 + 线程在tile内的列偏移
        int col = col_base + i * TILE_SIZE + tx;
        // 计算在当前tile内的行索引
        int k_idx = ty;
        
        // 边界检查
        if (col < K && (0 * TILE_SIZE + k_idx) < N) {
            // 将B矩阵的数据加载到共享内存Bs的load_buf_idx缓冲区
            // Bs[buffer][行索引][列索引] = B[全局行][全局列]
            Bs[load_buf_idx][ty][i * TILE_SIZE + tx] = B[(0 * TILE_SIZE + k_idx) * K + col];
        } else {
            // 越界位置填充0
            Bs[load_buf_idx][ty][i * TILE_SIZE + tx] = 0.0f;
        }
    }

    // 确保所有线程都完成了初始数据加载
    __syncthreads();

    // 切换缓冲区索引，为第一次计算做准备
    // 初始加载到了缓冲区0，所以下次加载到缓冲区1，本次计算使用缓冲区0
    load_buf_idx = 1;
    calc_buf_idx = 0;

    // ==================== 主计算循环（双缓冲核心）====================
    // 循环处理所有的tiles，每个tile大小为TILE_SIZE
    for (int t = 0; t < num_tiles; t++) {
        
        // 1. 启动下一次迭代的数据加载（如果不是最后一次迭代）
        //    这部分与当前迭代的计算并行执行，实现计算和内存访问的重叠
        if (t < num_tiles - 1) {
            // 加载下一个A矩阵tile到load_buf_idx缓冲区
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                // 计算全局行索引
                int row = row_base + i * TILE_SIZE + ty;
                // 当前列索引
                int k_idx = tx;
                // 计算下一个tile在N维度的起始位置
                int next_n_tile = (t + 1) * TILE_SIZE;
                
                // 边界检查并加载数据
                if (row < M && (next_n_tile + k_idx) < N) {
                    As[load_buf_idx][i * TILE_SIZE + ty][tx] = 
                        A[row * N + next_n_tile + k_idx];
                } else {
                    As[load_buf_idx][i * TILE_SIZE + ty][tx] = 0.0f;
                }
            }

            // 加载下一个B矩阵tile到load_buf_idx缓冲区
            #pragma unroll
            for (int i = 0; i < TK; i++) {
                // 计算全局列索引
                int col = col_base + i * TILE_SIZE + tx;
                // 当前行索引
                int k_idx = ty;
                // 计算下一个tile在N维度的起始位置
                int next_n_tile = (t + 1) * TILE_SIZE;
                
                // 边界检查并加载数据
                if (col < K && (next_n_tile + k_idx) < N) {
                    Bs[load_buf_idx][ty][i * TILE_SIZE + tx] = 
                        B[(next_n_tile + k_idx) * K + col];
                } else {
                    Bs[load_buf_idx][ty][i * TILE_SIZE + tx] = 0.0f;
                }
            }
        }

        // 2. 执行当前迭代的计算，使用calc_buf_idx指向的缓冲区
        //    计算矩阵乘法：val[i][j] += sum_k(As[...][k] * Bs[k][...])
        //    这是标准的矩阵乘法计算：C[i][j] += A[i][k] * B[k][j]
        for (int k = 0; k < TILE_SIZE; k++) {      // 遍历当前tile的K维度(0-15)
            for (int i = 0; i < TM; i++) {         // 遍历结果矩阵的行(0-7)
                for (int j = 0; j < TK; j++) {     // 遍历结果矩阵的列(0-7)
                    // 累积计算：val[i][j] += A元素 × B元素
                    // As[当前计算缓冲区][线程负责的第i行 + 偏移][k维度索引]
                    // Bs[当前计算缓冲区][k维度索引][线程负责的第j列 + 偏移]
                    val[i][j] += 
                        As[calc_buf_idx][i * TILE_SIZE + ty][k] * 
                        Bs[calc_buf_idx][k][j * TILE_SIZE + tx];
                }
            }
        }

        // 3. 等待当前迭代的计算和下一次迭代的加载都完成
        //    确保所有线程都完成了计算和数据加载，避免数据竞争
        __syncthreads();

        // 4. 交换缓冲区索引：为下一次迭代做准备
        //    下次计算使用刚才加载的缓冲区，下次加载使用刚才计算的缓冲区
        int tmp = load_buf_idx;
        load_buf_idx = calc_buf_idx;
        calc_buf_idx = tmp;
    }

    // ==================== 结果写回全局内存 ====================
    // 将每个线程计算的TM×TK结果写回到全局内存C矩阵
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TK; j++) {
            // 计算结果在全局矩阵C中的位置
            int row = row_base + i * TILE_SIZE + ty;  // 全局行索引
            int col = col_base + j * TILE_SIZE + tx;  // 全局列索引
            
            // 边界检查并写回结果
            if (row < M && col < K) {
                C[row * K + col] = val[i][j];  // C[行][列] = 计算结果
            }
        }
    }
}

// 主函数：矩阵乘法入口点
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // 配置线程块大小：16×16 = 256个线程
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    // 配置网格大小：覆盖整个结果矩阵C(M×K)
    // blocksPerGrid.x = CEIL(K, BK)：需要多少个block来覆盖K维度
    // blocksPerGrid.y = CEIL(M, BM)：需要多少个block来覆盖M维度
    dim3 blocksPerGrid(CEIL(K, BK), CEIL(M, BM));
    
    // 启动CUDA内核
    matmul_double_buffer<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    
    // 等待内核执行完成
    cudaDeviceSynchronize();
}