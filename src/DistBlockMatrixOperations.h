#include "DistBlockMatrix.h"
#include "Vector.h"

#ifndef DIST_BLOCK_MATRIX_OPERATIONS_H_
#define DIST_BLOCK_MATRIX_OPERATIONS_H_

/*
 *  Calculates column means and stores it in Vector col_means.
 */
int
DistBlockMatrix_column_means(DistBlockMatrix *mat,
                             Vector *col_means);

int
DistBlockMatrix_normalize(DistBlockMatrix *mat);

#endif
