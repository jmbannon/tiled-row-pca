#include "BlockMatrixOperations.h"
#include "BlockMatrix.h"
#include "BlockOperations.h"
#include "Vector.h"
#include "error.h"
#include "lapacke.h"
#include <cblas.h>
#include <stdio.h>

double*
BlockMatrix_get_block(BlockMatrix *mat,
                      int blk_i,
                      int blk_j)
{
    return &mat->data[POS(blk_i * BLK_LEN, blk_j * BLK_LEN, mat->nr_blk_cols)];
}


int
BlockMatrix_column_sums(BlockMatrix *mat,
                        Vector *col_sums,
                        double scalar)
{
    if (col_sums->nr_elems != mat->nr_cols) {
        return INVALID_DIMS;
    }

    double *mat_blk;
    double *vec_blk;
    for (int blk_col = 0; blk_col < mat->nr_blk_cols; blk_col++) {
        vec_blk = Vector_get_block(col_sums, blk_col);
        for (int blk_row = 0; blk_row < mat->nr_blk_rows; blk_row++) {
            mat_blk = BlockMatrix_get_block(mat, blk_row, blk_col);
            Block_col_sums(mat_blk, vec_blk);
        }
    }

    // col_sums *= scalar
    cblas_dscal(col_sums->nr_elems, scalar, col_sums->data, 1);
    return 0;
}

