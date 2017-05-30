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
// int Test_TileQR_dgeqt2()
// {
//     int res;

//     int m = 5;
//     int n = 5;

//     /////////////////////////////////////////////////


//     Matrix A, T, A_orig, I;
//     cublasHandle_t handle;

//     res = cublasCreate(&handle);
//     CHECK_ERROR_RETURN(res != CUBLAS_STATUS_SUCCESS, "Failed to create cublas handle", 1);

//     res = Matrix_init(&A, m, n);
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

//     res = Matrix_init_rand_device(&A, m, n, 253L);
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

//     res = Matrix_init_rand_device(&A_orig, m, n, 253L);
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

//     res = Matrix_init(&T, n, n);
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

//     res = Matrix_init_zero_device(&T, n, n);
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

//     res = Matrix_init_diag_device(&I, m, m, 1.0);
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

//     res = Matrix_copy_device_to_host(&A_orig);
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

//     printf("Before QR:\n");
//     Matrix_print(&A_orig);

//     Block_dgeqt2(&handle, &A, &T);

//     res = Matrix_copy_device_to_host(&A);
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

//     printf("R with Householder vectors:\n");
//     Matrix_print(&A);

//     res = Matrix_copy_device_to_host(&T);
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

//     printf("T matrix:\n");
//     Matrix_print(&T);

//     // Form Q using householder vectors and T
//     //
//     // Q = I + (Y * T * t(Y))

//     // Calculates T = T * t(Y)
//     Numeric alpha = 1.0;
//     #if FLOAT_NUMERIC
//         res = cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, n, n, &alpha, A.data_d, m, T.data_d, n, T.data_d, n);
//     #else
//         res = cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, n, n, &alpha, A.data_d, m, T.data_d, n, T.data_d, n);
//     #endif
//     CHECK_CUBLAS_RETURN(res, "Failed to compute T = T * t(Y)");

//     // Calculates T = Y * T
//     #if FLOAT_NUMERIC
//         res = cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, &alpha, A.data_d, m, T.data_d, n, T.data_d, n);
//     #else
//         res = cublasDtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, &alpha, A.data_d, m, T.data_d, n, T.data_d, n);
//     #endif
//     CHECK_CUBLAS_RETURN(res, "Failed to compute T = Y * T");

//     // Calculates T = Q = T + I
//     #if FLOAT_NUMERIC
//         res = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, T.data_d, n, &alpha, I.data_d, n, T.data_d, n);
//     #else
//         res = cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, T.data_d, n, &alpha, I.data_d, n, T.data_d, n);
//     #endif
//     CHECK_CUBLAS_RETURN(res, "Failed to compute T = Q = T + I");

//     // Calculates T = QR
//     #if FLOAT_NUMERIC
//         res = cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, &alpha, A.data_d, n, T.data_d, n, T.data_d, n);
//     #else
//         res = cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, &alpha, A.data_d, n, T.data_d, n, T.data_d, n);
//     #endif
//     CHECK_CUBLAS_RETURN(res, "Failed to compute T = QR");

//     res = Matrix_copy_device_to_host(&T);
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

//     printf("After QR:\n");
//     printf("A:\n");
//     Matrix_print(&A_orig);
//     printf("QR:\n");
//     Matrix_print(&T);

//     bool equals = false;
//     printf("TODO: Test by assembling Q matrix and multiply QR, compare with A\n");

//     return equals ? 0 : 1;
// }


int Test_TileQR_dgeqt2_internal(int m, int n)
{
    int res;

    /////////////////////////////////////////////////
    printf("begin\n");

    Matrix A, T, Q, A_orig, I;
    cublasHandle_t handle;

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

    res = Matrix_init(&T, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

    res = Matrix_init_zero_device(&T, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_init_zero_device(&Q, m, m);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_init(&Q, m, m);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

    res = Matrix_init_diag_device(&I, m, n, 1.0);
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
    res = TileQR_cublasDgemm_mht(CUBLAS_DIAG_UNIT, n, m, n, alpha, T.data_d, n, A.data_d, m, Q.data_d, m);
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = T * t(Y)");



    res = Matrix_copy_device_to_host(&Q);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy Q matrix from device to host");

    printf("T * t(Y)\n");
    Matrix_print(&Q);


    // Calculates T = Y * T
    TileQR_cublasDgemm_hmn(CUBLAS_DIAG_UNIT, m, n, n, alpha, A.data_d, m, Q.data_d, n, Q.data_d, m);
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = Y * T");

    res = Matrix_copy_device_to_host(&Q);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy Q matrix from device to host");

    printf("Y * T\n");
    Matrix_print(&Q);

    // Calculates T = Q = T + I
    #if FLOAT_NUMERIC
        res = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, &alpha, Q.data_d, m, &alpha, I.data_d, m, Q.data_d, m);
    #else
        res = cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, &alpha, Q.data_d, m, &alpha, I.data_d, m, Q.data_d, m);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = Q = T + I");

    // Calculates T = QR
    #if FLOAT_NUMERIC
        res = cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha, A.data_d, n, Q.data_d, m, Q.data_d, m);
    #else
        res = cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha, A.data_d, n, Q.data_d, m, Q.data_d, m);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute T = QR");

    res = Matrix_copy_device_to_host(&Q);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy Q matrix from device to host");

    printf("After QR:\n");
    printf("A:\n");
    Matrix_print(&A_orig);
    printf("QR:\n");
    Matrix_print(&Q);

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
    printf("\n\n");
    return Test_TileQR_dgeqt2_internal(10, 5);
}