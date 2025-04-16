#include <stdio.h>

__global__ void mykernel(int *data){

  (*data)++;
}

