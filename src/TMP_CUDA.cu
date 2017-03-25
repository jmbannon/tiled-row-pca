#include "TMP_CUDA.h"
#include <cuda.h>

__global__ void mykernel(void) {}

extern "C"
void someFunction(void) {
    mykernel<<<1,1>>>();
}
