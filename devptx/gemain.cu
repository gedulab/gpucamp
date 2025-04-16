#include <stdio.h>

__global__ void mykernel(int *data);

int main(){

  int *d_data, h_data = 0;
  cudaMalloc((void **)&d_data, sizeof(int));
  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  mykernel<<<1,1>>>(d_data);
  cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
  printf("data = %d\n", h_data);
  return 0;
}
