#include "../TileQR_Operations.h"
#include "../error.h"
#include "../BlockMatrix.h"
#include "../Matrix.h"
#include "../MatrixOperations.h"
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

    res = Matrix_init_constant(&expected_output, m, n, 3.0);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant matrix");

    res = Matrix_qr_r(&expected_output);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute lapack QR on matrix");

    res = BlockMatrix_init_constant(&A, m, n, 3.0);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant block matrix");

    res = BlockMatrix_init_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init block matrix on device");

    res = BlockMatrix_copy_host_to_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy block matrix from host to device");

    res = BlockMatrix_TileQR_single_thread(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute TileQR on block matrix");

    res = BlockMatrix_copy_device_to_host(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy block matrix from device to host");

    printf("tile qr\n");
    BlockMatrix_print(&A);
    printf("\nlapack\n");
    Matrix_print(&expected_output);

    bool equals = false;
    return equals ? 0 : 1;
}

int Test_TileQR_16_16() {
    return Test_TileQR(16, 8);
}