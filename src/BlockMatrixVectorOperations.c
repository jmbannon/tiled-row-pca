#include "BlockMatrixVectorOperations.h"
#include "BlockMatrix.h"
#include "BlockMatrixOperations.h"
#include "BlockOperations.h"
#include "Vector.h"
#include "error.h"

int
BlockMatrixVector_sub(BlockMatrix *mat,
                      Vector *vec)
{
    int res;
    double *mat_blk;
    double *vec_blk;
    
    if (mat->nr_cols != vec->nr_elems) {
        return INVALID_DIMS;   
    }
    
    for (int blk_row = 0; blk_row < mat->nr_blk_rows; blk_row++) {
        for (int blk_col = 0; blk_col < mat->nr_blk_cols; blk_col++) {
            vec_blk = Vector_get_block(vec, blk_col);
            mat_blk = BlockMatrix_get_block(mat, blk_row, blk_col);
            Block_sub_vec(mat_blk, vec_blk);
        }
    }
    return 0;
}
