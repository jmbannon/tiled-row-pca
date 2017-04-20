#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Block.h"
#include "error.h"
#include "constants.h"

int
Block_init(Numeric **blk)
{
    *blk = (Numeric *)malloc(BLK_SIZE * sizeof(Numeric));
    CHECK_MALLOC_RETURN(*blk);
    return 0;
}

int
Block_init_zero(Numeric **blk)
{
    *blk = (Numeric *)calloc(BLK_SIZE, sizeof(Numeric));
    CHECK_MALLOC_RETURN(*blk);
    return 0;
}

int
Block_init_seq(Numeric **blk)
{
    int res = Block_init(blk);
    CHECK_ZERO_RETURN(res);
    for (int i = 0; i < BLK_SIZE; i++) {
        (*blk)[i] = (Numeric)i;
    }
    return 0;
}

int
Block_init_diag(Numeric **blk)
{
    int res = Block_init_zero(blk);
    CHECK_ZERO_RETURN(res);
    for (int i = 0; i < BLK_SIZE; i++) {
        (*blk)[GET_BLK_POS(i, i)] = 1.0;
    }
    return 0;
}


int
Block_get_elem(Numeric *blk, int i, int j, Numeric *data)
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
Block_print(Numeric *blk)
{
    for (int i = 0; i < BLK_LEN; i++) {
        for (int j = 0; j < BLK_LEN; j++) {
            printf("%.7f, ", blk[GET_BLK_POS(i, j)]);
        }
        printf("\n");
    }
}

static int
_Block_zero_lower_tri_diag(Numeric *blk)
{
    for (int j = 0; j < BLK_LEN; j++) {
        for (int i = j; i < BLK_LEN; i++) {
            blk[GET_BLK_POS(i, j)] = 0;
        }
    }
    return 0;
}

static int
_Block_zero_lower_tri(Numeric *blk)
{
    for (int j = 0; j < BLK_LEN; j++) {
        for (int i = j+1; i < BLK_LEN; i++) {
            blk[GET_BLK_POS(i, j)] = 0;
        }
    }
    return 0;
}

static int
_Block_zero_upper_tri_diag(Numeric *blk)
{
    for (int j = 0; j < BLK_LEN; j++) {
        for (int i = 0; i <= j; i++) {
            blk[GET_BLK_POS(i, j)] = 0;
        }
    }
    return 0;
}

static int
_Block_zero_upper_tri(Numeric *blk)
{
    for (int j = 0; j < BLK_LEN; j++) {
        for (int i = 0; i < j; i++) {
            blk[GET_BLK_POS(i, j)] = 0;
        }
    }
    return 0;
}

int
Block_zero_tri(Numeric *blk, bool upper, bool diag)
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

