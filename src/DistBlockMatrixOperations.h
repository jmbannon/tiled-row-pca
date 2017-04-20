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
DistBlockMatrix_device_column_means(DistBlockMatrix *mat,
                                    Vector *col_means);

/**
 * Normalizes a matrix by subtracting each column by its respective column means.
 * @param mat Matrix to normalize.
 */
int
DistBlockMatrix_global_normalize(DistBlockMatrix *mat);

// TODO: Move this to a vector operation class
int
DistBlockMatrix_host_global_column_means(DistBlockMatrix *mat,
                                         Vector *local_col_means,
                                         Vector *global_col_means);

#endif
