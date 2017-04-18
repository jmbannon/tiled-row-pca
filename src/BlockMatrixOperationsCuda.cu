#include "BlockMatrixOperations.h"
#include "BlockMatrix.h"
#include "Vector.h"
#include "constants.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixColumnSumsKernel(double *in, double *out, int nrBlkCols, double scalar)
{
	__shared__ double localColSum;

	// Each thread handles a single Block column
	int col = blockIdx.x;

	// Each thread will start at the top of a Block
	int row = BLK_LEN * threadIdx.y;

	// Index of the input
	int idx = POS(row, col, nrBlkCols);
	int idxMax = idx + BLK_LEN;

	double blockColSum = 0.0;
	for (; idx < idxMax; idx++) {
		blockColSum += in[idx];
	}

	if (row == 0) {
		localColSum = 0;
	}
	__syncthreads();
	atomicAdd(&localColSum, blockColSum);
	__syncthreads();

	if (row == 0) {
		out[col] = scalar * localColSum;
	}
}

extern "C"
int CudaBlockMatrix_column_sums(BlockMatrix *in, double *d_in, Vector *out, double scalar)
{
	double *d_out = NULL;
	const int outSize = in->nr_blk_cols * BLK_LEN * sizeof(double);

	cudaMalloc((void **)&d_out, outSize);

	dim3 dimGrid(in->nr_cols, 1);
	dim3 dimBlock(1, in->nr_blk_rows);
	
    matrixColumnSumsKernel<<<dimGrid, dimBlock>>>(d_in, d_out, in->nr_blk_cols, scalar);
    cudaMemcpy(out->data, d_out, outSize, cudaMemcpyDeviceToHost);

    return 0;
}

extern "C"
int BlockMatrix_column_sums(BlockMatrix *in, Vector *out, double scalar)
{
	int nrBlks = in->nr_blk_rows * in->nr_blk_cols;
	double *d_in;

	const int inSize = nrBlks * BLK_SIZE * sizeof(double);
	cudaMalloc((void **)&d_in, inSize);

	cudaMemcpy(d_in, in->data, inSize, cudaMemcpyHostToDevice);
	return CudaBlockMatrix_column_sums(in, d_in, out, scalar);
}



