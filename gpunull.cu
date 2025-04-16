#include <stdio.h>

__global__ void gpu_hello(int* ptr){
    printf("Hi GPU! I will Keng you;-:\n");
    *ptr = 0x666;
    printf("Survived after pointer %p dereference.\n", ptr);
}

int main() {    
    gpu_hello<<<1,1>>>((int*)0x888); 

    cudaDeviceSynchronize();
    return 0;
}
