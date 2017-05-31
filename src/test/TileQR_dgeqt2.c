#include "../BlockQROperations.h"
#include "../error.h"
#include "../Vector.h"
#include "../Matrix.h"
#include "DoubleCompare.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdbool.h>

int Test_TileQR_dgeqt2_internal(int m, int n)
{
    int res;

    Matrix A, T, Q, A_orig, I;
    cublasHandle_t handle;

    /////////////////////////////////////////////////

    res = cublasCreate(&handle);
    CHECK_ERROR_RETURN(res != CUBLAS_STATUS_SUCCESS, "Failed to create cublas handle", 1);

    res = Matrix_init(&A, m, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

    res = Matrix_init_rand_device(&A, m, n, 253L);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_init_rand_device(&A_orig, m, n, 253L);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_init(&A_orig, m, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_copy_device_to_host(&A_orig);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

    res = Matrix_init(&T, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

    res = Matrix_init_zero_device(&T, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_init_zero_device(&Q, m, m);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_init(&Q, m, m);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

    res = Matrix_init(&I, m, m);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

    res = Matrix_init_diag_device(&I, m, m, 1.0);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Block_dgeqt2(&handle, &A, &T);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute TileQR helper function dgeqt2");

    res = Matrix_copy_device_to_host(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");



    // Form Q using householder vectors and T
    //
    // Q = I + (Y * T * t(Y))
    //

    // Calculates T = T * t(Y)
    Numeric alpha = 1.0;
    res = TileQR_cublasDgemm_mht(CUBLAS_DIAG_UNIT, n, m, n, alpha, T.data_d, n, A.data_d, m, Q.data_d, m);
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = T * t(Y)");

    // Calculates T = Y * T
    TileQR_cublasDgemm_hmn(CUBLAS_DIAG_UNIT, m, m, n, alpha, A.data_d, m, Q.data_d, m, I.data_d, m);
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = Y * T");

    // Calculates T = Q = T + I
    res = Matrix_add_diag_device(&I, 1.0);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to add identity matrix");

    // Calculates T = QR
    #if FLOAT_NUMERIC
        res = cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha, A.data_d, m, I.data_d, m, Q.data_d, m);
    #else
        res = cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha, A.data_d, m, I.data_d, m, Q.data_d, m);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = QR");

    res = Matrix_copy_device_to_host(&Q);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy Q matrix from device to host");

    bool equals = true;
    for (int i = 0; i < (m * n); i++) {
        equals = DoubleCompare(A_orig.data[i], Q.data[i]);
        if (!equals) {
            break;
        }
    }

    return equals ? 0 : 1;
}

int Test_TileQR_dgeqt2() {
    return Test_TileQR_dgeqt2_internal(5, 5);
}

int Test_TileQR_dgeqt2_rect() {
    return Test_TileQR_dgeqt2_internal(10, 5);
}