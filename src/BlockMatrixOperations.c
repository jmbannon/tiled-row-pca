#include "BlockMatrixOperations.h"
#include "BlockMatrix.h"
#include "BlockOperations.h"
#include "Vector.h"
#include "error.h"
#include "lapacke.h"
#include "constants.h"
#include <cblas.h>
#include <stdio.h>

Numeric*
BlockMatrix_get_block(BlockMatrix *mat,
                      int blk_i,
                      int blk_j)
{
    return &mat->data[POS(blk_i * BLK_LEN, blk_j * BLK_LEN, mat->nr_blk_cols)];
}
