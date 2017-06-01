#include "../BlockMatrixOperations.h"
#include "../BlockMatrix.h"
#include "../Vector.h"
#include "../constants.h"
#include "../error.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void device_column_sums(Numeric *in, Numeric *out, int nrBlkCols, Numeric scalar)
{
	__shared__ Numeric localColSum;

	// Each thread handles a single Block column
	int col = blockIdx.x;

	// Each thread will start at the top of a Block
	int row = BLK_LEN * threadIdx.y;

	// Index of the input
	int idx = POS(row, col, nrBlkCols);
	int idxMax = idx + BLK_LEN;

	Numeric blockColSum = 0.0;
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
int BlockMatrix_device_column_sums(BlockMatrix *in, Vector *out, Numeric scalar)
{
    dim3 dimGrid(in->nr_cols, 1);
    dim3 dimBlock(1, in->nr_blk_rows);
    
    device_column_sums<<<dimGrid, dimBlock>>>(in->data_d, out->data_d, in->nr_blk_cols, scalar);
    return 0;
}


