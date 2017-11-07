#include "Perf.h"
#include "../TileQR_Operations.h"
#include "../error.h"
#include "../BlockMatrix.h"
#include "../Matrix.h"
#include "../MatrixOperations.h"
#include "../Timer.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <stdio.h>
#include <stdbool.h>

int CudaQR(int m, int n, double *runtime_ms)
{
    int res, ress;
    int lwork = 0;

    cusolverDnHandle_t handle;
    Matrix A;
    Timer t;

    Numeric *work;
    Numeric *tau;

    int *resp;
    res = cudaMalloc((void **)&resp, sizeof(int));
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init resp");

    res = cusolverDnCreate(&handle);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to create handle");

    res = Matrix_init_constant_device(&A, m, n, 253.360);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init rand block matrix");

    #if FLOAT_NUMERIC
        res = cusolverDnSgeqrf_bufferSize(handle, m, n, A.data_d, m, &lwork);
    #else
        res = cusolverDnDgeqrf_bufferSize(handle, m, n, A.data_d, m, &lwork);
    #endif
    CHECK_ZERO_ERROR_RETURN(res, "Failed to get buffer size");

    res = cudaMalloc((void **)&work, lwork * sizeof(Numeric));
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init work array");

    int min = m > n ? n : m;
    res = cudaMalloc((void **)&tau, min * sizeof(Numeric));
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init tau");

    Timer_start(&t);

    #if FLOAT_NUMERIC
        res = cusolverDnSgeqrf(handle, m, n, A.data_d, m, tau, work, lwork, resp);
    #else
        res = cusolverDnDgeqrf(handle, m, n, A.data_d, m, tau, work, lwork, resp);
    #endif

    ress = cudaDeviceSynchronize();

    CHECK_CUSOLVER_RETURN(res, "Failed to compute QR");
    CHECK_CUSOLVER_RETURN(ress, "Failed to synchronize");

    Timer_end(&t);

    *runtime_ms = Timer_dur_sec(&t);

    res = Matrix_free_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to free block matrix from device");

    return 0;
}

int TileQR(int m, int n, double *runtime_ms)
{
    int res;

    BlockMatrix A;
    Timer t;

    res = BlockMatrix_init_rand(&A, m, n, 500, 253);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init rand block matrix");

    res = BlockMatrix_init_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init block matrix on device");

    res = BlockMatrix_copy_host_to_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy block matrix from host to device");

    cudaProfilerStart();

    Timer_start(&t);
    res = BlockMatrix_TileQR_multi_thread(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute TileQR on block matrix");
    Timer_end(&t);

    *runtime_ms = Timer_dur_sec(&t);

    cudaProfilerStop();

    res = BlockMatrix_free(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to free block matrix from host");

    res = BlockMatrix_free_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to free block matrix from device");

    return 0;
}

int TileQR_1k_4k(double *runtime_ms) {
    return TileQR(1000, 4000, runtime_ms);
}

int CudaQR_1k_4k(double *runtime_ms) {
    return CudaQR(1000, 4000, runtime_ms);
}

int TileQR_2k_4k(double *runtime_ms) {
    return TileQR(2000, 4000, runtime_ms);
}

int CudaQR_2k_4k(double *runtime_ms) {
    return CudaQR(2000, 4000, runtime_ms);
}

int TileQR_4k_4k(double *runtime_ms) {
    return TileQR(4000, 4000, runtime_ms);
}

int CudaQR_4k_4k(double *runtime_ms) {
    return CudaQR(4000, 4000, runtime_ms);
}

int TileQR_8k_4k(double *runtime_ms) {
    return TileQR(8000, 4000, runtime_ms);
}

int CudaQR_8k_4k(double *runtime_ms) {
    return CudaQR(8000, 4000, runtime_ms);
}

int TileQR_16k_4k(double *runtime_ms) {
    return TileQR(16000, 4000, runtime_ms);
}

int CudaQR_16k_4k(double *runtime_ms) {
    return CudaQR(16000, 4000, runtime_ms);
}



int TileQR_4k_1k(double *runtime_ms) {
    return TileQR(4000, 1000, runtime_ms);
}

int CudaQR_4k_1k(double *runtime_ms) {
    return CudaQR(4000, 1000, runtime_ms);
}

int TileQR_4k_2k(double *runtime_ms) {
    return TileQR(4000, 2000, runtime_ms);
}

int CudaQR_4k_2k(double *runtime_ms) {
    return CudaQR(4000, 2000, runtime_ms);
}

int TileQR_4k_8k(double *runtime_ms) {
    return TileQR(4000, 8000, runtime_ms);
}

int CudaQR_4k_8k(double *runtime_ms) {
    return CudaQR(4000, 8000, runtime_ms);
}

int TileQR_4k_16k(double *runtime_ms) {
    return TileQR(4000, 16000, runtime_ms);
}

int CudaQR_4k_16k(double *runtime_ms) {
    return CudaQR(4000, 16000, runtime_ms);
}

