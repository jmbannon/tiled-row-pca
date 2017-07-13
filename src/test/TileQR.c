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

bool compare_r(Matrix *a, BlockMatrix *b) {
    Numeric a_sign, b_sign, compare_sign;
    
    int min_dim = (a->nr_rows > a->nr_cols) ? a->nr_cols : a->nr_rows;

    for (int i = 0; i < min_dim; i++) {
        // Values are unique up to the sign of the rows of R.
        a_sign = (a->data[MAT_POS(i, i, a->nr_rows)] >= 0) ? 1 : -1;
        b_sign = (b->data[POS(i, i, b->nr_blk_cols)] >= 0) ? 1 : -1;
        compare_sign = (a_sign == b_sign) ? 1 : -1;

        for (int j = i; j < a->nr_cols; j++) {
            if (!DoubleCompare(a->data[MAT_POS(i, j, a->nr_rows)], compare_sign * b->data[POS(i, j, b->nr_blk_cols)])) {
                return false;
            }
        }
    }
    return true;
}

int Test_TileQR(int m, int n, int range, unsigned int seed)
{
    int res;

    BlockMatrix A;
    Matrix expected_output;

    res = BlockMatrix_init_rand(&A, m, n, range, seed);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init rand block matrix");

    res = Matrix_init(&expected_output, m, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant matrix");

    res = Matrix_copy_BlockMatrix(&expected_output, &A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy BlockMatrix into Matrix");

    res = Matrix_qr_r(&expected_output);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute lapack QR on matrix");

    res = BlockMatrix_init_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init block matrix on device");

    res = BlockMatrix_copy_host_to_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy block matrix from host to device");

    res = BlockMatrix_TileQR_single_thread(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute TileQR on block matrix");

    res = BlockMatrix_copy_device_to_host(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy block matrix from device to host");

    bool equals = compare_r(&expected_output, &A);

    res = BlockMatrix_free(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to free block matrix from host");

    res = BlockMatrix_free_device(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to free block matrix from device");

    res = Matrix_free(&expected_output);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to free matrix from host");

    return equals ? 0 : 1;
}

int Test_TileQR_16_16() {
    return Test_TileQR(16, 8, 4, 360);
}

int Test_TileQR_1024_64() {
    return Test_TileQR(1024, 64, 500, 253);
}

int Test_TileQR_71_29() {
    return Test_TileQR(79, 29, 500, 666);
}