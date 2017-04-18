#include "BlockMatrix.h"
#include "constants.h"
#include <cuda.h>
#include <cuda_runtime.h>

// TODO: check return values of CUDA functions
extern "C"
int
BlockMatrix_to_device(BlockMatrix *in, double **d_in)
{
    int nrBlks = in->nr_blk_rows * in->nr_blk_cols;

    const int inSize = nrBlks * BLK_SIZE * sizeof(double);
    cudaMalloc((void **)d_in, inSize);

    cudaMemcpy(*d_in, in->data, inSize, cudaMemcpyHostToDevice);

    return 0;
}