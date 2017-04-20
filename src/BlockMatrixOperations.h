#include "BlockMatrix.h"
#include "Vector.h"
#include "constants.h"

#ifndef _BLOCK_MATRIX_OPERATIONS_H
#define _BLOCK_MATRIX_OPERATIONS_H

Numeric*
BlockMatrix_get_block(BlockMatrix *mat,
                      int blk_i,
                      int blk_j);

/**
  * Computes column sums on the BlockMatrix and stores it in the output vector,
  * all within the device.
  *
  * @param in BlockMatrix with data in device.
  * @param out Vector to store output in device.
  * @param scalar Scalar to multiply the column sums by.
  *
  * @return 0 on success, errno on error.
  */
#ifdef __cplusplus
extern "C"
#endif
int BlockMatrix_device_column_sums(BlockMatrix *in, Vector *out, Numeric scalar);

#endif
