#include "Perf.h"
#include "../TileQR_Operations.h"
#include "../error.h"
#include "../BlockMatrix.h"
#include "../Matrix.h"
#include "../MatrixOperations.h"
#include "../Timer.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <stdio.h>
#include <stdbool.h>

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

int TileQR_1000_1000(double *runtime_ms) {
    return TileQR(1024, 256, runtime_ms);
}