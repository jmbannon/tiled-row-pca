#include "BlockMatrix.h"
#include "Block.h"
#include "error.h"
#include "Vector.h"
#include <stdbool.h>
#include "lapacke.h"
#include "constants.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * Sets the matrix's dimension meta-info.
 */
static void
BlockMatrix_set_dimensions(BlockMatrix *mat,
                           int nr_rows,
                           int nr_cols)
{
    mat->nr_rows = nr_rows;
    mat->nr_cols = nr_cols;
    mat->nr_blk_rows = nr_rows / BLK_LEN + (nr_rows % BLK_LEN != 0);
    mat->nr_blk_cols = nr_cols / BLK_LEN + (nr_cols % BLK_LEN != 0);
    mat->data = NULL;
    mat->data_d = NULL;
}

/**
 * Initializes a matrix's meta-info but does not allocate
 * memory for its data.
 */
int
BlockMatrix_init_info(BlockMatrix *mat,
                      int nr_rows,
                      int nr_cols)
{
    BlockMatrix_set_dimensions(mat, nr_rows, nr_cols);
    return 0;
}

#ifdef __cplusplus
extern "C"
#endif
int
BlockMatrix_size_bytes(BlockMatrix *mat)
{
    return mat->nr_blk_rows * mat->nr_blk_cols * BLK_SIZE * sizeof(double);
}

int
BlockMatrix_init(BlockMatrix *mat,
                 int nr_rows,
                 int nr_cols)
{
    BlockMatrix_set_dimensions(mat, nr_rows, nr_cols);
    int size = mat->nr_blk_rows * mat->nr_blk_cols * BLK_SIZE;
    mat->data = (double *)malloc(size * sizeof(double));
    CHECK_MALLOC_RETURN(mat->data);

    return 0;
}

/**
 * Initialize a matrix with the specified constant.
 */
int
BlockMatrix_init_constant(BlockMatrix *mat,
                          int nr_rows,
                          int nr_cols,
                          double constant)
{
    int res = BlockMatrix_init(mat, nr_rows, nr_cols);
    CHECK_ZERO_RETURN(res);

    int size = mat->nr_blk_rows * mat->nr_blk_cols * BLK_SIZE;
    for (int i = 0; i < size; i++) {
        mat->data[i] = constant;
    }

    return 0;
}

/**
 * Initializes a matrix with 0s.
 */
int
BlockMatrix_init_zero(BlockMatrix *mat,
                      int nr_rows,
                      int nr_cols)
{
    BlockMatrix_set_dimensions(mat, nr_rows, nr_cols);
    int size = mat->nr_blk_rows * mat->nr_blk_cols * BLK_SIZE;

    mat->data = (double *)calloc(size, sizeof(double));
    CHECK_MALLOC_RETURN(mat->data);

    return 0;
}

int
BlockMatrix_print(BlockMatrix *mat)
{
    int idx;
    int i, j;
    for (i = 0; i < mat->nr_rows; i++) {
        for (j = 0; j < mat->nr_cols; j++) {
            idx = POS(i,j,mat->nr_blk_cols);
            printf("%.3f ", mat->data[idx]);
        }
        printf(" %d %d \n", i, j);
    }
    return 0;
}

void
BlockMatrix_print_blocks(BlockMatrix *mat)
{
    const int max_width = 7;
    int idx;
    for (int i = 0; i < mat->nr_rows; i++) {
        for (int j = 0; j < mat->nr_cols; j++) {
            idx = POS(i, j, mat->nr_blk_cols);
            printf("%*.3f ", max_width, mat->data[idx]);
            if (j % BLK_LEN == (BLK_LEN - 1)) {
                printf("  ");
            }
        }
        printf("\n");
        if (i % BLK_LEN == (BLK_LEN - 1)) {
            printf("\n");
        }
    }
}

void
BlockMatrix_print_padding(BlockMatrix *mat)
{
    int idx;
    int i, j;
    for (i = 0; i < mat->nr_blk_rows * BLK_LEN; i++) {
        for (j = 0; j < mat->nr_blk_cols * BLK_LEN; j++) {
            idx = POS(i,j,mat->nr_blk_cols);
            printf("%.3f ", mat->data[idx]);
        }
        printf(" %d %d \n", i, j);
    }
}

int
BlockMatrix_free(BlockMatrix *mat)
{
    if (mat->data != NULL) {
        free(mat->data);
    }
    return 0;
}

