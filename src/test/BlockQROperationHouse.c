#include "../BlockQROperations.h"
#include "../error.h"
#include "../Vector.h"
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

    Numeric *I;
    Numeric *P;

    cublasHandle_t cublas_handle;

    res = cublasCreate(&cublas_handle);
    CHECK_ERROR_RETURN(res != CUBLAS_STATUS_SUCCESS, "Failed to create cublas handle", 1);

    res = Vector_init_constant(&x, n, constant);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant");

    res = Vector_init_device(&x);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant on device");

    res = Vector_init_constant(&v, n, constant);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant");

    res = Vector_init_device(&v);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant on device");

    res = Vector_copy_host_to_device(&x);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy vector from host to device");

    res = Block_house(&cublas_handle, &x, &v);
    CHECK_ZERO_ERROR_RETURN(res, "Block householder helper function failed");


    bool equals = false;
    return equals ? 0 : 1;
}