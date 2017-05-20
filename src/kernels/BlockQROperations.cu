#include "../BlockQROperations.h"
#include "../constants.h"
#include "../error.h"
#include "../Vector.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

/**
  * Given an n-vector x, computes an n-vector v with v[0] = 1 such that (I - 2*v*t(v) / t(v) * v)x is 
  * zero in all but the first component.
  * @param x n-vector used to compute v.
  * @param v n-vector to store output
  * @param n Length of vectors x and v
  *
  * @see{https://www.youtube.com/watch?v=d-yPM-bxREs}
  */
__device__ void house(cublasHandle_t *handle, Numeric *x, Numeric *v, int n)
{
    cublasHandle_t handle2;
    int res = cublasCreate(&handle2);

    Numeric x_norm;
    res = 0;

    #if FLOAT_NUMERIC
    	res = cublasScopy(handle2, n, x, 1, v, 1);
    	res = cublasSnrm2(handle2, n, x, 1, &x_norm);
    #else
    	res = cublasDcopy(handle2, n, x, 1, v, 1);
    	res = cublasDnrm2(handle2, n, x, 1, &x_norm);
    #endif

    if (x_norm != 0) {
    	const Numeric sign = x[0] >= 0 ? 1.0 : -1.0;
    	const Numeric beta = 1.0 / (x[0] + (sign * x_norm));
    	#if FLOAT_NUMERIC
    		cublasSscal(handle2, n - 1, &beta, &v[1], 1);
    	#else
    		cublasDscal(handle2, n - 1, &beta, &v[1], 1);
    	#endif
    }
    v[0] = 1.0;
}

__global__ void Block_house_kernel(cublasHandle_t *handle, Numeric *x, Numeric *v, int n) {
    house(handle, x, v, n);
}

extern "C"
int
Block_house(cublasHandle_t *handle, Vector *in, Vector *out) {    
    Block_house_kernel<<<1, 1>>>(handle, in->data_d, out->data_d, in->nr_elems);
    return 0;
}