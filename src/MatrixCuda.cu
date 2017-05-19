#include "Matrix.h"
#include "constants.h"
#include "error.h"
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Copies data from host to device.
 */
extern "C"
int
Matrix_copy_host_to_device(Matrix *in)
{
    int res = cudaMemcpy(in->data_d, in->data, Matrix_size_bytes(in), cudaMemcpyHostToDevice);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
Matrix_copy_device_to_host(Matrix *in)
{
    int res = cudaMemcpy(in->data, in->data_d, Matrix_size_bytes(in), cudaMemcpyDeviceToHost);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

/**
 * CudaMalloc device matrix.
 */
extern "C"
int
Matrix_init_device(Matrix *in)
{
    int res = cudaMalloc((void **)&in->data_d, Matrix_size_bytes(in));
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
Matrix_free_device(Matrix *in)
{
    int res = cudaFree(in->data_d);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}