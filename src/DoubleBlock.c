#include "DoubleBlock.h"
#include "constants.h"
#include "error.h"
#include <stdlib.h>

/**
 * Inits a 2*BLK_LEN x 2*BLK_LEN column-wise matrix
 */
inline int
DoubleBlock_init(double **dbl_blk)
{
    *dbl_blk = (double *)malloc(4 * BLK_SIZE * sizeof(double));
    CHECK_MALLOC_RETURN(*dbl_blk);
    return 0;
}

inline int
DoubleBlock_init_diag(double **dbl_blk)
{
    int res = DoubleBlock_init(dbl_blk);
    CHECK_ZERO_RETURN(res);
    for (int i = 0; i < 2*BLK_LEN; i++) {

    }
}
