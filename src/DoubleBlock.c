#include "DoubleBlock.h"
#include "constants.h"
#include "error.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int
DoubleBlock_init(Numeric **dbl_blk)
{
    *dbl_blk = (Numeric *)malloc(4 * BLK_SIZE_MEM);
    CHECK_MALLOC_RETURN(*dbl_blk);
    return 0;
}

int
DoubleBlock_init_diag(Numeric **dbl_blk)
{
    int res = DoubleBlock_init(dbl_blk);
    CHECK_ZERO_RETURN(res);
    for (int i = 0; i < 2*BLK_LEN; i++) {

    }
}

int
DoubleBlock_init_rbind(Numeric **rbind, Numeric *top, Numeric *bot)
{
    *rbind = (Numeric *)malloc(2 * BLK_SIZE_MEM);
    CHECK_MALLOC_RETURN(*rbind);

    for (int j = 0; j < BLK_LEN; j++) {
        memcpy(&(*rbind)[j * (BLK_LEN * 2)], &top[j * BLK_LEN], BLK_LEN_MEM);
        memcpy(&(*rbind)[j * (BLK_LEN * 2) + BLK_LEN], &bot[j * BLK_LEN], BLK_LEN_MEM);
    }
    return 0;
}

void
DoubleBlock_print(Numeric *dbl_blk)
{
    for (int i = 0; i < BLK_LEN * 2; i++) {
        for (int j = 0; j < BLK_LEN; j++) {
            printf("%.7f, ", dbl_blk[(j * BLK_LEN * 2) + i]);
        }
        printf("\n");
    }
}

int
DoubleBlock_free(Numeric **dbl_blk)
{
    free(*dbl_blk);
}
