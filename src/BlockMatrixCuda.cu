#include "BlockMatrix.h"
#include "constants.h"
#include "error.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C"
int
BlockMatrix_copy_host_to_device(BlockMatrix *in)
{
    int res = cudaMemcpy(in->data_d, in->data, BlockMatrix_size_bytes(in), cudaMemcpyHostToDevice);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
BlockMatrix_copy_device_to_host(BlockMatrix *in)
{
    int res = cudaMemcpy(in->data, in->data_d, BlockMatrix_size_bytes(in), cudaMemcpyDeviceToHost);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
BlockMatrix_init_device(BlockMatrix *in)
{
    int res = cudaMalloc((void **)&in->data_d, BlockMatrix_size_bytes(in));
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
BlockMatrix_free_device(BlockMatrix *in)
{
    int res = cudaFree(in->data_d);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}