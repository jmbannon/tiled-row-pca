#include "../TileQR_Operations.h"
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
  * Tests HouseHolder Matrix helper function house
  * 
  * Given a n-vector x, computes an n-vector v with v[0] = 1 such that
  * P = I - 2*v*t(v) / t(v)*v, Px is zero in all but the first component
  */
int Test_BlockQROperationHouse()
{
    int res;
    int n = 10;

    const Numeric constant = 5.5;
    const Numeric expectedOutput = 0.0;

    /////////////////////////////////////////////////

    Vector x;
    Vector v;

    Matrix P;

    cublasHandle_t cublas_handle;

    res = cublasCreate(&cublas_handle);
    CHECK_ERROR_RETURN(res != CUBLAS_STATUS_SUCCESS, "Failed to create cublas handle", 1);

    res = Vector_init_constant(&x, n, constant);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant");

    res = Vector_init_device(&x);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant on device");

    res = Vector_copy_host_to_device(&x);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy vector from host to device");


    res = Vector_init_constant(&v, n, constant);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init vector");

    res = Vector_init_device(&v);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant on device");

    res = TileQR_house(&x, &v);
    CHECK_ZERO_ERROR_RETURN(res, "Block householder helper function failed");

    res = Matrix_init_diag_device(&P, n, n, 1.0);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_init_constant(&P, n, n, 2.0);
    CHECK_ZERO_ERROR_RETURN(res, "failed to init constant matrix on host");

    res = Matrix_copy_device_to_host(&P);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from host to device");

    res = Vector_copy_device_to_host(&v);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy vector from host to device");

    Numeric v_norm = 0;
    #if FLOAT_NUMERIC
        res = cublasSnrm2(cublas_handle, n, v.data_d, 1, &v_norm);
    #else
        res = cublasDnrm2(cublas_handle, n, v.data_d, 1, &v_norm);
    #endif

    Numeric v_norm_squared = v_norm * v_norm;

    // Need to compute
    // P = I - 2*v*t(v) / t(v)*v, we know ||v||^2 = t(v)*v
    //   = I - 2*v*t(v) / ||v||^2
    //   = (-2 / ||v||^2) v*t(v) + I
    //   = aAB + bI
    Numeric alpha = -2.0 / v_norm_squared;
    Numeric beta = 1.0;
    int m = n;
    int k = 1;
    // int n = n

    #if FLOAT_NUMERIC
        res = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, v.data_d, n, v.data_d, 1, &beta, P.data_d, n);
    #else
        res = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, v.data_d, n, v.data_d, 1, &beta, P.data_d, n);
    #endif

    res = Matrix_copy_device_to_host(&P);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy output P matrix from device to host");

    // Need to compute
    // v = Px
    //   = aPx + 0b
    alpha = 1.0;
    beta = 0.0;
    #if FLOAT_NUMERIC
        cublasSgemv(cublas_handle, CUBLAS_OP_N, n, n, &alpha, P.data_d, n, x.data_d, 1, &beta, v.data_d, 1);
    #else
        cublasDgemv(cublas_handle, CUBLAS_OP_N, n, n, &alpha, P.data_d, n, x.data_d, 1, &beta, v.data_d, 1);
    #endif

    res = Vector_copy_device_to_host(&v);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy vector from device to host");

    // Assures only the first element is non-zero while the rest are zero.
    bool equals = !DoubleCompare(v.data[0], 0.0);
    for (int i = 1; i < v.nr_elems; i++) {
        if (!DoubleCompare(v.data[i], 0.0)) {
            equals = false;
            break;
        }
    }

    Vector_free(&x);
    Vector_free_device(&x);
    Vector_free(&v);
    Vector_free_device(&v);
    Matrix_free(&P);
    Matrix_free_device(&P);

    return equals ? 0 : 1;
}