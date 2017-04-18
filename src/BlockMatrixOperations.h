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

#endif
