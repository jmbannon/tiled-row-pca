#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Block.h"
#include "error.h"
#include "constants.h"

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
Block_init_diag(double **blk)
{
    int res = Block_init_zero(blk);
    CHECK_ZERO_RETURN(res);
    for (int i = 0; i < BLK_SIZE; i++) {
        (*blk)[GET_BLK_POS(i, i)] = 1.0;
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

static int
_Block_zero_lower_tri_diag(double *blk)
{
    for (int j = 0; j < BLK_LEN; j++) {
        for (int i = j; i < BLK_LEN; i++) {
            blk[GET_BLK_POS(i, j)] = 0;
        }
    }
    return 0;
}

static int
_Block_zero_lower_tri(double *blk)
{
    for (int j = 0; j < BLK_LEN; j++) {
        for (int i = j+1; i < BLK_LEN; i++) {
            blk[GET_BLK_POS(i, j)] = 0;
        }
    }
    return 0;
}

static int
_Block_zero_upper_tri_diag(double *blk)
{
    for (int j = 0; j < BLK_LEN; j++) {
        for (int i = 0; i <= j; i++) {
            blk[GET_BLK_POS(i, j)] = 0;
        }
    }
    return 0;
}

static int
_Block_zero_upper_tri(double *blk)
{
    for (int j = 0; j < BLK_LEN; j++) {
        for (int i = 0; i < j; i++) {
            blk[GET_BLK_POS(i, j)] = 0;
        }
    }
    return 0;
}

int
Block_zero_tri(double *blk, bool upper, bool diag)
{
    int res;
    if (upper) {
        if (diag) {
            res = _Block_zero_upper_tri_diag(blk);
        } else {
            res = _Block_zero_upper_tri(blk);
        }
    } else {
        if (diag) {
            res = _Block_zero_lower_tri_diag(blk);
        } else {
            res = _Block_zero_lower_tri(blk);
        }
    }
    return res;
}

void
Block_print_rbind(double *rbind)
{
    for (int i = 0; i < BLK_LEN * 2; i++) {
        for (int j = 0; j < BLK_LEN; j++) {
            printf("%.7f, ", rbind[(j * BLK_LEN * 2) + i]);
        }
        printf("\n");
    }
}

