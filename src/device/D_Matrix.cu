#include "../Matrix.h"
#include "../constants.h"
#include "../error.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
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
	in[blockIdx.x] = constant;
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
    
    device_init_constant<<<nr_rows * nr_cols, 1>>>(mat->data_d, constant);
    return 0;
}

__global__ void device_add_diag(Numeric *in, int nr_rows, Numeric constant)
{
	int pos = MAT_POS(blockIdx.x, blockIdx.x, nr_rows);
	in[pos] += constant;
}

extern "C"
int
Matrix_add_diag_device(Matrix *mat,
			           Numeric constant)
{
	device_add_diag<<<mat->nr_cols, 1>>>(mat->data_d, mat->nr_rows, constant);
	return 0;
}

extern "C"
int
Matrix_init_diag_device(Matrix *mat,
			            int nr_rows,
			            int nr_cols,
			            Numeric constant)
{
	int res = Matrix_init_constant_device(mat, nr_rows, nr_cols, 0.0);
	CHECK_ZERO_RETURN(res);

	res = Matrix_add_diag_device(mat, constant);
	return res;
}

__global__ void device_init_seq_mem(Numeric *in, int nr_rows)
{
	int idx = MAT_POS(blockIdx.x, blockIdx.y, nr_rows);
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

    dim3 dimGrid(nr_rows, nr_cols);

	device_init_seq_mem<<<dimGrid, 1>>>(mat->data_d, nr_rows);
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



__global__ void device_init_rand(Numeric *in, int nr_rows, unsigned long seed)
{
	curandState state;
	int idx = MAT_POS(blockIdx.x, blockIdx.y, nr_rows);

	curand_init(seed, idx, 0, &state);
	in[idx] = curand_uniform(&state);
}

extern "C"
int
Matrix_init_rand_device(Matrix *mat,
			            int nr_rows,
			            int nr_cols,
			            unsigned long seed)
{
	int res = Matrix_init_device(mat, nr_rows, nr_cols);
	CHECK_ZERO_RETURN(res);

    dim3 dimGrid(nr_rows, nr_cols);

	device_init_rand<<<dimGrid, 1>>>(mat->data_d, nr_rows, seed);
	return 0;
}