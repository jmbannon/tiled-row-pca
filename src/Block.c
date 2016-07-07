#include <stdlib.h>
#include "Block.h"
#include "error.h"

int
Block_init(Block blk)
{
    blk = malloc(BLK_SIZE * sizeof(double));
    CHECK_MALLOC_RETURN(blk);
    return 0;
}

int
Block_init_zero(Block blk)
{
    blk = calloc(BLK_SIZE, sizeof(double));
    CHECK_MALLOC_RETURN(blk);
    return 0;
}

int
Block_get_elem(Block blk, int i, int j, double *data)
{
    const int pos = GET_BLK_POS(i, j);
    if (pos < 0 || pos > BLK_SIZE) {
        return INVALID_INDICES;
    } else {
        *data = blk[pos];
        return 0;
    }   
}
