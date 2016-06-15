#include "BlockMatrixOperations.h"
#include "BlockMatrix.h"
#include "Vector.h"

static double*
BlockMatrix_get_block(BlockMatrix *mat,
                      int blk_i,
                      int blk_j)
{
    return &mat->data[POS(blk_i * BLK_LEN, blk_j * BLK_LEN, mat->nr_blk_cols)];
}

static void
BlockMatrix_block_col_sums(double *block,
                           double *col_sum)
{
    int row_offset;
    for (int i = 0; i < BLK_LEN; i++) {
        row_offset = i * BLK_LEN;
        for (int j = 0; j < BLK_LEN; j++) {
            col_sum[j] += block[row_offset + j]; 
        }
    }
}

int
BlockMatrix_column_sums(BlockMatrix *mat,
                        Vector *col_sums,
                        double scalar)
{
    if (mat->nr_rows <= 0) {
        return 0;
    }

    double *blk;
    double block_sum[BLK_LEN];
    int col_offset;
    for (int blk_row = 0; blk_row < mat->nr_blk_rows; blk_row++) {
        for (int blk_col = 0; blk_col < mat->nr_blk_cols; blk_col++) {
            blk = BlockMatrix_get_block(mat, blk_row, blk_col);
            col_offset = blk_col * BLK_LEN;
            BlockMatrix_block_col_sums(blk, &col_sums->data[col_offset]);
        }
    }
    for (int i = 0; i < col_sums->nr_elems; i++) {
        col_sums->data[i] *= scalar;
    }
    return 0;
}
