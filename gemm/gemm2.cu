#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#define GM_FLAG_SHARED     1
#define GM_FLAG_CPU_SERIAL 2
#define GM_FLAG_DOUBLE     4

unsigned int g_flags = GM_FLAG_DOUBLE;

int N = 2048; // 默认矩阵大小
int tile_width = 16; // 线程块大小

// CUDA核函数，用于矩阵乘法
__global__ void matrixMulSimple(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int i = 0; i < N; ++i) {
            value += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}
__global__ void matrixMulShared(float* A, float* B, float* C, int N, int tile_width) {
    //__shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    //__shared__ float sB[TILE_WIDTH][TILE_WIDTH];
    extern __shared__ float s[];
    float *sA = s;
    float *sB = s + tile_width * tile_width;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * tile_width + ty;
    int col = bx * tile_width + tx;

    float Pvalue = 0;

    for (int m = 0; m < N / tile_width; ++m) {
        sA[ty*tile_width+ tx] = A[row * N + m * tile_width + tx];
        sB[ty*tile_width+ tx] = B[(m * tile_width + ty) * N + col];

        __syncthreads();

        for (int k = 0; k < tile_width; ++k) {
            Pvalue += sA[ty*tile_width + k] * sB[k*tile_width + tx];
        }

        __syncthreads();
    }

    C[row * N + col] = Pvalue;
}
__global__ void matrixMulSimpleDouble(double *A, double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double value = 0;
        for (int i = 0; i < N; ++i) {
            value += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}
__global__ void matrixMulSharedDouble(double* A, double* B, double* C, int N, int tile_width) {
    // __shared__ double sA[TILE_WIDTH][TILE_WIDTH];
    // __shared__ double sB[TILE_WIDTH][TILE_WIDTH];
    extern __shared__ double d[];
    double *sA = d;
    double *sB = d + tile_width * tile_width;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * tile_width + ty;
    int col = bx * tile_width + tx;

    double Pvalue = 0;

    for (int m = 0; m < N / tile_width; ++m) {
        sA[ty*tile_width+tx] = A[row * N + m * tile_width + tx];
        sB[ty*tile_width+tx] = B[(m * tile_width + ty) * N + col];

        __syncthreads();

        for (int k = 0; k < tile_width; ++k) {
            Pvalue += sA[ty*tile_width + k] * sB[k*tile_width + tx];
        }

        __syncthreads();
    }

    C[row * N + col] = Pvalue;
}

// CPU矩阵乘法函数串行
void matrixMulCpuSerial(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// CPU矩阵乘法函数 - OpenMP
int omp_threads;
void matrixMulCpuOMP(float* A, float* B, float* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    omp_threads = omp_get_max_threads();   
}
// CPU矩阵乘法函数串行
void matrixMulCpuSerialDouble(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// CPU矩阵乘法函数 - OpenMP
void matrixMulCpuOMPDouble(double* A, double* B, double* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    omp_threads = omp_get_max_threads();   
}

int usage(const char* msg) {
    if(msg) {
        fprintf(stderr, "%s\n", msg);
    }
    fprintf(stderr, "Usage: %s [-s]\n", "gemm2");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -c      calc by CPU in serial mode\n");
    fprintf(stderr, "  -n <N>  specify matrix size\n");
    fprintf(stderr, "  -f      float type, default doubel\n");
    fprintf(stderr, "  -s      Use shared memory\n");
    fprintf(stderr, "  -t <w>  Tile width\n");
    return -1;
}
int gm_parse_cmdline(int argc, const char* argv[]) {
    for(int i = 1; i < argc; i++) {
        switch(argv[i][1]) {
            case 's': // shared
                g_flags |= GM_FLAG_SHARED;
                break;
            case 'c':// argc>1 && strcmp(argv[1],"-serial")==0
                g_flags |= GM_FLAG_CPU_SERIAL;
                break;
            case 'f':
                g_flags &= ~GM_FLAG_DOUBLE;
                break;
            case 'n':
                N = atoi(argv[++i]);
                break;
            case 't':
                tile_width = atoi(argv[++i]);
                break;
            default:
                return -1;
        }
    }
    return 0;
}
// 主函数
int main(int argc, const char* argv[]) 
{
    int ret = 0;
    double *h_A, *h_B, *h_C_CPU, *h_C_GPU; // 主机上的矩阵
    double *d_A, *d_B, *d_C; // 设备上的矩阵
    size_t size;
    bool diff = true;

    if(argc > 1) {
        ret = gm_parse_cmdline(argc, argv);
        if(ret < 0) {
            return usage("bad arguments");
        }
    }

    size = (g_flags & GM_FLAG_DOUBLE)? N * N * sizeof(double):N * N *sizeof(float);

    // 分配主机内存
    h_A = (double*)malloc(size);
    h_B = (double*)malloc(size);
    h_C_CPU = (double*)malloc(size);
    h_C_GPU = (double*)malloc(size);

    // 初始化矩阵A和B
    for (int i = 0; i < N * N; i++) {
        if(g_flags & GM_FLAG_DOUBLE){
            h_A[i] = 1.0;
            h_B[i] = 2.0;
        }
        else {
            ((float*)h_A)[i] = 1.0;
            ((float*)h_B)[i] = 2.0;
        }
    }

    // 分配设备内存
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义CUDA线程和块的数量
    dim3 threadsPerBlock(tile_width, tile_width);
    //dim3 threadsPerBlock(256, 256); // will fail on launch
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 创建事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start, 0);

    // 调用CUDA核函数
    if(g_flags & GM_FLAG_SHARED) {
        int size_shared = tile_width*tile_width*2;
        if(g_flags & GM_FLAG_DOUBLE) {
            size_shared *= sizeof(double);
            matrixMulSharedDouble<<<numBlocks, threadsPerBlock, size_shared>>>(d_A, d_B, d_C, N, tile_width);
        }
        else {
            size_shared *= sizeof(float);
            matrixMulShared<<<numBlocks, threadsPerBlock, size_shared>>>((float*)d_A, (float*)d_B, (float*)d_C, N, tile_width);
        }
    }
    else {
        if(g_flags & GM_FLAG_DOUBLE)
            matrixMulSimpleDouble<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        else
            matrixMulSimple<<<numBlocks, threadsPerBlock>>>((float*)d_A, (float*)d_B, (float*)d_C, N);
    }
    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(h_C_GPU, d_C, size, cudaMemcpyDeviceToHost);

    // 打印执行时间
    printf("CUDA Elapsed time: %f ms\n", elapsedTime);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if(g_flags & GM_FLAG_CPU_SERIAL) {
        cudaEventRecord(start, 0);

        // 在CPU上执行矩阵乘法
        memset(h_C_CPU, 0, size);
        if(g_flags & GM_FLAG_DOUBLE)
            matrixMulCpuSerialDouble(h_A, h_B, h_C_CPU, N);
        else
            matrixMulCpuSerial((float*)h_A, (float*)h_B, (float*)h_C_CPU, N);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("CpuSerial Elapsed time: %f ms\n", elapsedTime);

        // 比较两种结果
        for (int i = 0; i < N * N; i++) {
            if(g_flags & GM_FLAG_DOUBLE)
                diff = h_C_CPU[i] != h_C_GPU[i];
            else
                diff = ((float*)h_C_CPU)[i] != ((float*)h_C_GPU)[i];

            if (diff) {
                ret = -1;
                printf("Error: Result mismatch at index %d %lf != %lf\n", i, h_C_CPU[i], h_C_GPU[i]);
                break;
            }
        }
        if(ret == 0)
            printf("Results match.\n");
    }
    // 记录开始时间
    cudaEventRecord(start, 0);

    // 在CPU上执行矩阵乘法
    memset(h_C_CPU, 0, size);
    if(g_flags & GM_FLAG_DOUBLE)
        matrixMulCpuOMPDouble(h_A, h_B, h_C_CPU, N);
    else
        matrixMulCpuOMP((float*)h_A, (float*)h_B, (float*)h_C_CPU, N);
    
    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算执行时间
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // 打印执行时间
    printf("CpuOMP Elapsed time: %f ms using %d threads\n", elapsedTime, omp_threads);

    // 比较两种结果
    ret = 0;
    for (int i = 0; i < N * N; i++) {
        if(g_flags & GM_FLAG_DOUBLE)
            diff = h_C_CPU[i] != h_C_GPU[i];
        else
            diff = ((float*)h_C_CPU)[i] != ((float*)h_C_GPU)[i];
        if (diff) {
            ret = -1;
	        printf("Error: Result mismatch at index %d %lf != %lf\n", i, h_C_CPU[i], h_C_GPU[i]);
            break;
        }
    }
    if(ret == 0)
	    printf("Results match.\n");

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);

    return 0;
}

