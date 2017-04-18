#include "BlockMatrix.h"
#include "Vector.h"

#ifndef _BLOCK_MATRIX_OPERATIONS_H
#define _BLOCK_MATRIX_OPERATIONS_H

double*
BlockMatrix_get_block(BlockMatrix *mat,
                      int blk_i,
                      int blk_j);

#ifdef __cplusplus
extern "C"
#endif
int BlockMatrix_column_sums(BlockMatrix *in, Vector *out, double scalar);

#ifdef __cplusplus
extern "C"
#endif
int CudaBlockMatrix_column_sums(BlockMatrix *in, double *d_in, Vector *out, double scalar);

#ifdef __cplusplus
extern "C"
#endif
int CudaBlockMatrix_cuda_column_sums(BlockMatrix *in, double *d_in, double *d_out, double scalar);

#endif
