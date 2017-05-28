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

    Matrix A, T, Q, Q_;
    cublasHandle_t handle;

    /////////////////////////////////////////////////

    res = cublasCreate(&handle);
    CHECK_CUBLAS_RETURN(res, "Failed to create cublas handle");

    // Initialize random matrix A in device
    res = Matrix_init_rand_device(&A, m, n, 253L);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize random input matrix A in device");

    res = Matrix_init(&A, m, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize input matrix A in host");

    // Store original matrix A in host mem, manipulate in-place in device mem
    res = Matrix_copy_device_to_host(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy input matrix A from device to host");

    // Output matrix T from dgeqt2. Must be initialized as zero matrix since it is used in matrix-multiply
    res = Matrix_init_zero_device(&T, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize output matrix T in device");

    // m-by-m matrix Q, where QR = A
    res = Matrix_init_device(&Q, m, m);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize Q matrix in device");

    // m-by-n work matrix Q' used to calculate Q and stores output from QR (should equal A)
    res = Matrix_init(&Q_, m, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize work matrix Q' in host");

    res = Matrix_init_device(&Q_, m, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize work matrix Q' in device");

    /////////////////////////////////////////////////

    res = Block_dgeqt2(&handle, &A, &T);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute TileQR helper function dgeqt2");

    //
    // Form Q using householder vectors and T
    //
    // Q = I + (Y * T * t(Y))
    //
    ///////////////////////////////////////////////////
    Numeric alpha = 1.0;

    // Calculates Q' = Y * T
    TileQR_cublasDgemm_hmn(CUBLAS_DIAG_UNIT, m, n, n, alpha, A.data_d, m, T.data_d, n, Q_.data_d, m);
    CHECK_CUBLAS_RETURN(res, "Failed to compute Q' = Y * T");

    // Calculates Q = Q' * t(Y)
    //              = Y * T * t(Y)
    res = TileQR_cublasDgemm_mht(CUBLAS_DIAG_UNIT, m, m, n, alpha, Q_.data_d, m, A.data_d, m, Q.data_d, m);
    CHECK_CUBLAS_RETURN(res, "Failed to compute Q = Q' * t(Y)");

    // Calculates Q = I + Q
    //              = I + (Y * T * t(Y))
    res = Matrix_add_diag_device(&Q, 1.0);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to add identity matrix to Q");

    // Calculates Q' = QR
    //               = A
    #if FLOAT_NUMERIC
        res = cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha, A.data_d, m, Q.data_d, m, Q_.data_d, m);
    #else
        res = cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha, A.data_d, m, Q.data_d, m, Q_.data_d, m);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute Q' = QR");

    // Copy Q' to host, compare with original A stored in host memory
    res = Matrix_copy_device_to_host(&Q_);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy Q' matrix from device to host");

    bool equals = true;
    for (int i = 0; i < (m * n); i++) {
        equals = DoubleCompare(A.data[i], Q_.data[i]);
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