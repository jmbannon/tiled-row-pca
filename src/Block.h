#ifndef _BLOCK_H
#define _BLOCK_H

#include <stdbool.h>

/**
 * Inits a BLK_LEN x BLK_LEN column-wise matrix.
 * @param dbl_blk Block to malloc.
 */
int
Block_init(double **blk);

/**
 * Inits a BLK_LEN x BLK_LEN column-wise zero-matrix.
 * @param dbl_blk Block to calloc.
 */
int
Block_init_zero(double **blk);

/**
 * Inits a BLK_LEN x BLK_LEN column-wise sequential matrix.
 * Each value in the matrix is calculated by:
 * block[i,j] = i + (j * BLK_LEN)
 *
 * @param blk Block to allocate.
 */
int
Block_init_seq(double **blk);

/**
 * Prints a Block in matrix format.
 */
void
Block_print(double *blk);

/**
 * Retrieves a single value from a Block.
 * @param blk Block to retrieve value from.
 * @param i Row value.
 * @param j Column value.
 * @param data Pointer to fill with data.
 */
int
Block_get_elem(double *blk, int i, int j, double *data);

/**
 * Zeros out the either the upper-right or bottom-left corner (triangle)
 * of a Block.
 * @param blk Block to zero out a triangle.
 * @param upper If true zero out the upper-right corner.
 *              If false zero out the bottom-left corner.
 * @param diag If true zero out the diagonal.
 *             If false do not zero the diagonal.
 */
int
Block_zero_tri(double *blk, bool upper, bool diag);

#endif
