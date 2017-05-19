#include "Vector.h"
#include "error.h"
#include "constants.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

extern "C"
int
Vector_copy_host_to_device(Vector *in)
{
    int res = cudaMemcpy(in->data_d, in->data, Vector_size_bytes(in), cudaMemcpyHostToDevice);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
Vector_copy_device_to_host(Vector *in)
{
    int res = cudaMemcpy(in->data, in->data_d, Vector_size_bytes(in), cudaMemcpyDeviceToHost);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
Vector_init_device(Vector *in)
{
    int res = cudaMalloc((void **)&in->data_d, Vector_size_bytes(in));
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
Vector_free_device(Vector *in)
{
    int res = cudaFree(in->data_d);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}