#ifndef _BLOCK_DIST_PCA_CONSTANTS_H
#define _BLOCK_DIST_PCA_CONSTANTS_H

typedef double Numeric;

/** Default Block Attributes
 *
 *  BLK_LEN   row/col length for block matrix
 *  BLK_SIZE  BLK_LEN^2
 *
 *  GET_BLK_POS(i, j) position in column-major linear memory
 */
#define BLK_LEN (4)
#define BLK_SIZE (16)

#define GET_BLK_POS(i, j) ((j) * BLK_LEN + (i))



/**************************************************************/



/** Double Block Attributes
 *
 * DBL_BLK_LEN   must be 2 * BLK_LEN
 * DBL_BLK_SIZE  DBL_BLK_LEN^2
 *
 * GET_DBL_BLK_POS(i, j) position in column-major linear memory
 */
#define DBL_BLK_LEN (8)
#define DBL_BLK_SIZE (64)

#define GET_DBL_BLK_POS(i, j) ((j) * DBL_BLK_LEN + (i))



/**************************************************************/



/** Cblas defines
 */
#define CBLAS_NO_TRANS CblasNoTrans
#define CBLAS_TRANS CblasTrans
#define CBLAS_CONJ_TRANS CblasConjTrans


#endif
