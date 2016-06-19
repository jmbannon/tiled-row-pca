#include "BlockMatrix.h"
#include "Vector.h"

#ifndef BLOCK_MATRIX_VECTOR_OPERATIONS_H_
#define BLOCK_MATRIX_VECTOR_OPERATIONS_H_

/**
 * Subtracts a vector from each row.
 *
 * @param mat Input Matrix
 * @param vec Vector to subtract from Input Matrix rows.
 */
int
BlockMatrixVector_sub(BlockMatrix *mat,
                      Vector *vec);
#endif
