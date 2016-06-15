#include "BlockMatrix.h"
#include "error.h"
#include "Vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cblas.h>

static int
BlockMatrix_init_zero_flag(BlockMatrix *mat,
                           int nr_rows,
                           int nr_cols,
                           bool init_data)
{
    mat->nr_rows = nr_rows;
    mat->nr_cols = nr_cols;
    mat->nr_blk_rows = nr_rows / BLK_LEN + (nr_rows % BLK_LEN != 0);
    mat->nr_blk_cols = nr_cols / BLK_LEN + (nr_cols % BLK_LEN != 0);
    if (init_data) {
        int nr_elements = mat->nr_blk_rows * mat->nr_blk_cols * BLK_SIZE;
        mat->data = (double *)calloc(nr_elements, sizeof(double));
        CHECK_MALLOC_RETURN(mat->data);
    }

    return 0;
}

int
BlockMatrix_init_zero(BlockMatrix *mat,
                      int nr_rows,
                      int nr_cols)
{
    return BlockMatrix_init_zero_flag(mat, nr_rows, nr_cols, true);
}

int
BlockMatrix_init_info(BlockMatrix *mat,
                      int nr_rows,
                      int nr_cols)
{
    return BlockMatrix_init_zero_flag(mat, nr_rows, nr_cols, false);
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

