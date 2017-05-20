#include "../BlockQROperations.h"
#include "../error.h"
#include "../Vector.h"
#include "../Matrix.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdbool.h>

int Test_Matrix_copy_device_to_host()
{
    int res;
    int n = 10;

    Matrix P;

    res = Matrix_init(&P, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init matrix");

    res = Matrix_init_zero_device(&P, n, n);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init identity matrix on device");

    res = Matrix_copy_device_to_host(&P);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy matrix from device to host")

	bool equals = true;
    for (int i = 0; i < n * n; i++) {
    	if (P.data[i] != 0) {
    		equals = false;
    	}
    }

    return equals ? 0 : 1;
}