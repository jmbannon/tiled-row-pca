#include <stdio.h>
#include <stdlib.h>
#include "Block.h"
#include "error.h"

int
Block_init(double **blk)
{
    *blk = (double *)malloc(BLK_SIZE * sizeof(double));
    CHECK_MALLOC_RETURN(*blk);
    return 0;
}

int
Block_init_zero(double **blk)
{
    *blk = (double *)calloc(BLK_SIZE, sizeof(double));
    CHECK_MALLOC_RETURN(*blk);
    return 0;
}

int
Block_init_seq(double **blk)
{
    int res = Block_init(blk);
    CHECK_ZERO_RETURN(res);
    for (int i = 0; i < BLK_SIZE; i++) {
        (*blk)[i] = (double)i;
    }
    return 0;
}

int
Block_get_elem(double *blk, int i, int j, double *data)
{
    const int pos = GET_BLK_POS(i, j);
    if (pos < 0 || pos > BLK_SIZE) {
        return INVALID_INDICES;
    } else {
        *data = blk[pos];
        return 0;
    }   
}

void
Block_print(double *blk)
{
    for (int i = 0; i < BLK_LEN; i++) {
        for (int j = 0; j < BLK_LEN; j++) {
            printf("%.7f, ", blk[GET_BLK_POS(i, j)]);
        }
        printf("\n");
    }
}
