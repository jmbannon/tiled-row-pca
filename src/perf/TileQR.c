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

    cudaProfilerStart();

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

    cudaProfilerStop();

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

int TileQR_20k_4k(double *runtime_ms) {
    return TileQR(30000, 1000, runtime_ms);
}

int CudaQR_20k_4k(double *runtime_ms) {
    return CudaQR(30000, 1000, runtime_ms);
}