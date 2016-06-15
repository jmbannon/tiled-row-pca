#include "BlockMatrix.h"
#include "Vector.h"

#ifndef _BLOCK_MATRIX_OPERATIONS_H
#define _BLOCK_MATRIX_OPERATIONS_H

int
BlockMatrix_column_sums(BlockMatrix *mat,
                        Vector *col_means,
                        double scalar);

#endif
