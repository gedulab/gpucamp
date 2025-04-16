#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

int N = 2048; // 默认矩阵大小
int tile_width = 16; // 线程块大小

__device__ uint get_smid(void) {
     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}
__device__ uint get_warpid(void) {
     uint ret;
     asm("mov.u32 %0, %warpid;" : "=r"(ret) );
     return ret;
}
__device__ uint get_laneid(void) {
     uint ret;
     asm("mov.u32 %0, %laneid;" : "=r"(ret) );
     return ret;
}
// CUDA核函数，用于矩阵乘法
__global__ void diverse(int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row%2 == 0 && col%2 == 0) {
        printf("thread (%d,%d,%d) of block (%d,%d,%d) is calculating even cell %d,%d on SM%d warp%d lane%d\n",
          threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, row, col,
          get_smid(), get_warpid(), get_laneid());
    } else if(row%2 != 0 && col%2 != 0) {
	printf("thread (%d,%d,%d) of block (%d,%d,%d) is calculating odd cell  %d,%d on SM%d warp%d lane%d\n",
          threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, row, col,	
          get_smid(), get_warpid(), get_laneid());
    } else {
        printf("thread (%d,%d,%d) of block (%d,%d,%d) is calculating mixed cell %d,%d on SM%d warp%d lane%d\n",
          threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, row, col,
          get_smid(), get_warpid(), get_laneid());
    }

    printf("thread (%d,%d,%d) of block (%d,%d,%d) is quiting from SM%d\n",
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
        get_smid());
}

int usage(const char* msg) {
    if(msg) {
        fprintf(stderr, "%s\n", msg);
    }
    fprintf(stderr, "Usage: %s [-s]\n", "gesimt");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -n <N>  specify matrix size\n");
    fprintf(stderr, "  -t <w>  Tile width\n");
    return -1;
}
int gm_parse_cmdline(int argc, const char* argv[]) {
    for(int i = 1; i < argc; i++) {
        switch(argv[i][1]) {
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

    if(argc > 1) {
        ret = gm_parse_cmdline(argc, argv);
        if(ret < 0) {
            return usage("bad arguments");
        }
    }

    // 定义CUDA线程和块的数量
    dim3 threadsPerBlock(tile_width, tile_width);
    //dim3 threadsPerBlock(256, 256); // will fail on launch
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    diverse<<<numBlocks, threadsPerBlock>>>(N);
    cudaDeviceSynchronize();

    return 0;
}

