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
int Test_BlockQROperationHouseQR()
{
    int res;
    int n = 5;

    /////////////////////////////////////////////////


    Matrix A;
    cublasHandle_t cublas_handle;

    res = cublasCreate(&cublas_handle);
    CHECK_ERROR_RETURN(res != CUBLAS_STATUS_SUCCESS, "Failed to create cublas handle", 1);

    res = Matrix_init(&A, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to initialize matrix on host");

    res = Matrix_init_rand_device(&A, n, n, 253L);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_copy_device_to_host(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

    printf("Before QR:\n");
    Matrix_print(&A);

    Block_house_qr(&cublas_handle, &A);

    res = Matrix_copy_device_to_host(&A);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy constant matrix from device to host");

    printf("After QR:\n");
    Matrix_print(&A);

    bool equals = false;
    printf("TODO: Test by assembling Q matrix and multiply QR, compare with A\n");

    return equals ? 0 : 1;
}