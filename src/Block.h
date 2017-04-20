#include "constants.h"

#ifndef _BLOCK_H
#define _BLOCK_H

#include <stdbool.h>

/**
 * Inits a BLK_LEN x BLK_LEN column-wise matrix.
 * @param dbl_blk Block to malloc.
 */
int
Block_init(Numeric **blk);

/**
 * Inits a BLK_LEN x BLK_LEN column-wise zero-matrix.
 * @param dbl_blk Block to calloc.
 */
int
Block_init_zero(Numeric **blk);

/**
 * Inits a BLK_LEN x BLK_LEN column-wise sequential matrix.
 * Each value in the matrix is calculated by:
 * block[i,j] = i + (j * BLK_LEN)
 *
 * @param blk Block to allocate.
 */
int
Block_init_seq(Numeric **blk);

/**
 * Prints a Block in matrix format.
 */
void
Block_print(Numeric *blk);

/**
 * Retrieves a single value from a Block.
 * @param blk Block to retrieve value from.
 * @param i Row value.
 * @param j Column value.
 * @param data Pointer to fill with data.
 */
int
Block_get_elem(Numeric *blk, int i, int j, Numeric *data);

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
Block_zero_tri(Numeric *blk, bool upper, bool diag);

#endif
