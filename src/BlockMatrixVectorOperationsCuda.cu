#include "BlockMatrixVectorOperations.h"
#include "BlockMatrix.h"
#include "constants.h"
#include "BlockMatrixOperations.h"
#include "BlockOperations.h"
#include "Vector.h"
#include "error.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void device_sub(double *in, double *vec, int nrBlkCols)
{

	int row = BLK_LEN * blockIdx.y;
	int col = BLK_LEN * threadIdx.x;

	int idx = POS(row, col, nrBlkCols);
	int col_idx;
	for (int i = 0; i < BLK_LEN; i++) {
		col_idx = col + i;
		for (int j = 0; j < BLK_LEN; j++) {
			in[idx++] -= vec[col_idx];
		}
	}
}

extern "C"
int BlockMatrixVector_device_sub(BlockMatrix *in, Vector *vec)
{
    dim3 dimGrid(1, in->nr_blk_rows);
    dim3 dimBlock(in->nr_blk_cols, 1);
    
    device_sub<<<dimGrid, dimBlock>>>(in->data_d, vec->data_d, in->nr_blk_cols);
    return 0;
}