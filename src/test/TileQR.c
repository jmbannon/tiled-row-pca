#include "../TileQR_Operations.h"
#include "../error.h"
#include "../BlockMatrix.h"
#include "DoubleCompare.h"
#include "lapacke.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdbool.h>

int Test_TileQR(int m, int n)
{
    int res;

    BlockMatrix A;
    Matrix expected_output;

    res = BlockMatrix_init_constant(&A, m, n, 3.0);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant block matrix");

    res = Matrix_init_constant(&expected_output, m, n, 3.0);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant matrix");

    int min_dim = m > n ? n : m;
    Numeric *tau = malloc(min_dim * sizeof(Numeric));
    CHECK_MALLOC_RETURN(tau);

    Numeric work_size;
    Numeric *work;
    int work_query = -1;
    #if FLOAT_NUMERIC
        LAPACK_sgeqrf(&m, &n, expected_output.data, &m, tau, &work_size, &work_query, &res);
    #else
        LAPACK_dgeqrf(&m, &n, expected_output.data, &m, tau, &work_size, &work_query, &res);
    #endif
    CHECK_ZERO_ERROR_RETURN(res, "Failed to query size of array");

    int work_size_int = (int)work_size;
    work = malloc(work_size_int * sizeof(Numeric));
    CHECK_MALLOC_RETURN(work);

    printf("starting qr\n");

    #if FLOAT_NUMERIC
        LAPACK_sgeqrf(&m, &n, expected_output.data, &m, tau, work, &work_size_int, &res);
    #else
        LAPACK_dgeqrf(&m, &n, expected_output.data, &m, tau, work, &work_size_int, &res);
    #endif
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute lapack QR decomposition");

    res = BlockMatrix_init_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init block matrix on device");

    res = BlockMatrix_copy_host_to_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy block matrix from host to device");

    res = BlockMatrix_TileQR_single_thread(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute TileQR on block matrix");

    res = BlockMatrix_copy_device_to_host(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy block matrix from device to host");

    printf("tile qr\n");
    BlockMatrix_print_blocks(&A);
    printf("\nlapack\n");
    Matrix_print(&expected_output);

    bool equals = false;
    return equals ? 0 : 1;
}

int Test_TileQR_16_16() {
    return Test_TileQR(16, 8);
}