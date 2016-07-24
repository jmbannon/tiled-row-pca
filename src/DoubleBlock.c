#include "DoubleBlock.h"
#include "constants.h"
#include "error.h"
#include <stdlib.h>
#include <string.h>

/**
 * Inits a 2*BLK_LEN x 2*BLK_LEN column-wise matrix
 */
int
DoubleBlock_init(double **dbl_blk)
{
    *dbl_blk = (double *)malloc(4 * BLK_SIZE * sizeof(double));
    CHECK_MALLOC_RETURN(*dbl_blk);
    return 0;
}

int
DoubleBlock_init_diag(double **dbl_blk)
{
    int res = DoubleBlock_init(dbl_blk);
    CHECK_ZERO_RETURN(res);
    for (int i = 0; i < 2*BLK_LEN; i++) {

    }
}

int
DoubleBlock_init_rbind(double **rbind, double *top, double *bot)
{
    *rbind = (double *)malloc(BLK_SIZE * 2 * sizeof(double));
    CHECK_MALLOC_RETURN(*rbind);

    for (int j = 0; j < BLK_LEN; j++) {
        memcpy(&(*rbind)[j * (BLK_LEN * 2)], &top[j * BLK_LEN], BLK_LEN * sizeof(double));
        memcpy(&(*rbind)[j * (BLK_LEN * 2) + BLK_LEN], &bot[j * BLK_LEN], BLK_LEN * sizeof(double));
    }
    return 0;
}
