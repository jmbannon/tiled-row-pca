#include "Matrix.h"
#include "constants.h"
#include "error.h"
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Copies data from host to device.
 */
extern "C"
int
Matrix_copy_host_to_device(Matrix *in)
{
    int res = cudaMemcpy(in->data_d, in->data, Matrix_size_bytes(in), cudaMemcpyHostToDevice);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
Matrix_copy_device_to_host(Matrix *in)
{
    int res = cudaMemcpy(in->data, in->data_d, Matrix_size_bytes(in), cudaMemcpyDeviceToHost);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

/**
 * CudaMalloc device matrix.
 */
extern "C"
int
Matrix_init_device(Matrix *in,
				   int nr_rows,
				   int nr_cols)
{
	int res = Matrix_init_info(in, nr_rows, nr_cols);
	CHECK_ZERO_RETURN(res);

    res = cudaMalloc((void **)&in->data_d, Matrix_size_bytes(in));
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
Matrix_free_device(Matrix *in)
{
    int res = cudaFree(in->data_d);
    CHECK_SUCCESS_RETURN(res);

    return 0;
}

extern "C"
int
Matrix_init_constant_device(Matrix *mat,
	                        int nr_rows,
	                        int nr_cols,
	                        Numeric constant)
{
	int res = Matrix_init_device(mat, nr_rows, nr_cols);
    CHECK_ZERO_RETURN(res);

    int size = nr_rows * nr_cols;
    for (int i = 0; i < size; i++) {
        mat->data_d[i] = constant;
    }

    return 0;
}

extern "C"
int
Matrix_init_diag_device(Matrix *mat,
			            int nr_rows,
			            int nr_cols,
			            Numeric constant)
{
	int res = Matrix_init_zero_device(mat, nr_rows, nr_cols);
	CHECK_ZERO_RETURN(res);

	int min = nr_rows < nr_cols ? nr_rows : nr_cols;
	for (int i = 0; i < min; i++) {
		mat->data_d[MAT_POS(i, i, mat->nr_rows)] = 1.0;
	}

	return 0;
}

/**
 * Initializes a matrix with 0s.
 *
 * @param mat Matrix to initialize.
 * @param nr_rows Number of rows.
 * @param nr_cols Number of columns.
 */
extern "C"
int
Matrix_init_zero_device(Matrix *mat,
		                int nr_rows,
		                int nr_cols)
{
	return Matrix_init_constant_device(mat, nr_rows, nr_cols, 0.0);
}