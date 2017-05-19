#include <stdio.h>
#include "BlockOperations.h"
#include "BlockMatrix.h"
#include "error.h"
#include "constants.h"

void
Block_col_sums(Numeric *block,
               Numeric *col_sum)
{
    for (int i = 0; i < BLK_SIZE; i++) {
        col_sum[i / BLK_LEN] += block[i]; 
    }
}

/**
 * Subtracts block vector from block.
 */
void
Block_sub_vec(Numeric *block,
              Numeric *vec)
{
    for (int i = 0; i < BLK_SIZE; i++) {
        block[i] -= vec[i / BLK_LEN];
    }
}

