#include "Matrix.h"
#include "constants.h"
#include "error.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

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

__global__ void device_init_constant(Numeric *in, Numeric constant)
{
	in[threadIdx.x] = constant;
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

    dim3 dimGrid(1, 1);
    dim3 dimBlock(nr_rows * nr_cols, 1);
    
    device_init_constant<<<dimGrid, dimBlock>>>(mat->data_d, constant);
    return 0;
}

__global__ void device_init_diag(Numeric *in, int nr_rows, Numeric constant)
{
	int pos = MAT_POS(threadIdx.x, threadIdx.y, nr_rows);
	in[pos] = (threadIdx.x != threadIdx.y) ? 0.0 : constant;
}

extern "C"
int
Matrix_init_diag_device(Matrix *mat,
			            int nr_rows,
			            int nr_cols,
			            Numeric constant)
{
	int res = Matrix_init_device(mat, nr_rows, nr_cols);
	CHECK_ZERO_RETURN(res);

    dim3 dimBlock(nr_rows, nr_cols);

	device_init_diag<<<1, dimBlock>>>(mat->data_d, nr_rows, constant);
	return 0;
}

__global__ void device_init_seq_mem(Numeric *in, int nr_rows)
{
	int idx = MAT_POS(threadIdx.x, threadIdx.y, nr_rows);
	in[idx] = idx;
}

extern "C"
int
Matrix_init_seq_mem_device(Matrix *mat,
			               int nr_rows,
			               int nr_cols)
{
	int res = Matrix_init_device(mat, nr_rows, nr_cols);
	CHECK_ZERO_RETURN(res);

    dim3 dimBlock(nr_rows, nr_cols);

	device_init_seq_mem<<<1, dimBlock>>>(mat->data_d, nr_rows);
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