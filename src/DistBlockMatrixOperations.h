#include "DistBlockMatrix.h"
#include "Vector.h"

#ifndef DIST_BLOCK_MATRIX_OPERATIONS_H_
#define DIST_BLOCK_MATRIX_OPERATIONS_H_

/*
 * Calculates column means and stores it in Vector col_means.
 * @param mat Distributed matrix to calculate column means.
 * @param col_means Vector to store column means in.
 */
int
DistBlockMatrix_column_means(DistBlockMatrix *mat,
                             Vector *col_means);

/**
 * Normalizes a matrix by subtracting each column by its respective column means.
 * @param mat Matrix to normalize.
 */
int
DistBlockMatrix_normalize(DistBlockMatrix *mat);

#endif
