#include "constants.h"

#ifndef _DOUBLE_BLOCK_H_
#define _DOUBLE_BLOCK_H_

/**
 * Inits a 2*BLK_LEN x 2*BLK_LEN column-wise matrix.
 * @param dbl_blk DoubleBlock to malloc.
 */
int
DoubleBlock_init(Numeric **dbl_blk);

/**
 * Inits a 2*BLK_LEN x 2*BLK_LEN column-wise identity matrix.
 * @param dbl_blk DoubleBlock to allocate memory for.
 */
int
DoubleBlock_init_diag(Numeric **dbl_blk);

/**
 * Inits a 2*BLK_LEN x 2*BLK_LEN column-wsie matrix by row-binding
 * two exiting matrices.
 * @param rbind DoubleBlock to allocate memory for.
 * @param top Top Block to copy.
 * @param bot Bottom Block to copy.
 */
int
DoubleBlock_init_rbind(Numeric **rbind, Numeric *top, Numeric *bot);

/**
 * Prints a DoubleBlock in matrix format.
 */
void
DoubleBlock_print(Numeric *dbl_blk);

/**
 * Frees a DoubleBlock.
 */
int
DoubleBlock_free(Numeric **dbl_blk);

#endif
