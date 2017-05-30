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

/**
  * Tests HouseHolder Matrix helper function house_qr
  */
int Test_TileQR_dgeqt2()
{
    int res;
    int n = 5;

    /////////////////////////////////////////////////


    Matrix A, T, A_orig, I;
    cublasHandle_t handle;

    res = cublasCreate(&handle);
    CHECK_ERROR_RETURN(res != CUBLAS_STATUS_SUCCESS, "Failed to create cublas handle", 1);

    res = Matrix_init(&A, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

    res = Matrix_init_rand_device(&A, n, n, 253L);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_init_rand_device(&A_orig, n, n, 253L);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_init(&T, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

    res = Matrix_init_zero_device(&T, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_init(&T, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

    res = Matrix_init_diag_device(&I, n, n, 1.0);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_copy_device_to_host(&A_orig);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

    printf("Before QR:\n");
    Matrix_print(&A_orig);

    Block_dgeqt2(&handle, &A, &T);

    res = Matrix_copy_device_to_host(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

    printf("R with Householder vectors:\n");
    Matrix_print(&A);

    res = Matrix_copy_device_to_host(&T);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

    printf("T matrix:\n");
    Matrix_print(&T);

    // Form Q using householder vectors and T
    //
    // Q = I + (Y * T * t(Y))

    // Calculates T = T * t(Y)
    Numeric alpha = 1.0;
    #if FLOAT_NUMERIC
        res = cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, n, n, &alpha, A.data_d, n, T.data_d, n, T.data_d, n);
    #else
        res = cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, n, n, &alpha, A.data_d, n, T.data_d, n, T.data_d, n);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = T * t(Y)");

    // Calculates T = Y * T
    #if FLOAT_NUMERIC
        res = cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, &alpha, A.data_d, n, T.data_d, n, T.data_d, n);
    #else
        res = cublasDtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, &alpha, A.data_d, n, T.data_d, n, T.data_d, n);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = Y * T");

    // Calculates T = Q = T + I
    #if FLOAT_NUMERIC
        res = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, T.data_d, n, &alpha, I.data_d, n, T.data_d, n);
    #else
        res = cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, T.data_d, n, &alpha, I.data_d, n, T.data_d, n);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = Q = T + I");

    // Calculates T = QR
    #if FLOAT_NUMERIC
        res = cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, &alpha, A.data_d, n, T.data_d, n, T.data_d, n);
    #else
        res = cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, &alpha, A.data_d, n, T.data_d, n, T.data_d, n);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = QR");

    res = Matrix_copy_device_to_host(&T);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

    printf("After QR:\n");
    printf("A:\n");
    Matrix_print(&A_orig);
    printf("QR:\n");
    Matrix_print(&T);

    bool equals = false;
    printf("TODO: Test by assembling Q matrix and multiply QR, compare with A\n");

    return equals ? 0 : 1;
}