#include "Vector.h"

#ifndef _BLOCK_MATRIX_H
#define _BLOCK_MATRIX_H

#define BLK_LEN (4)
#define BLK_SIZE (BLK_LEN * BLK_LEN)

/* Translates a row,col index to a block format index */
#define POS(i,j,nr_blk_cols) \
((((i) / BLK_LEN) * (nr_blk_cols) * BLK_SIZE) + (((j) / BLK_LEN) * BLK_SIZE) + (((i) % BLK_LEN) * BLK_LEN) + ((j) % BLK_LEN))

typedef struct _BlockMatrix {
    int nr_rows;   // Number of rows
    int nr_cols;   // Number of columns

    int nr_blk_rows;   // Number of block rows
    int nr_blk_cols;   // Number of block columns

    double *data;
} BlockMatrix;

int
BlockMatrix_init_zero(BlockMatrix *mat,
                      int nr_rows,
                      int nr_cols);

int
BlockMatrix_init_info(BlockMatrix *mat,
                      int nr_rows,
                      int nr_cols);

int
BlockMatrix_column_sums(BlockMatrix *mat,
                        Vector *col_means,
                        double scalar);

int
BlockMatrix_print(BlockMatrix *mat);

void
BlockMatrix_print_blocks(BlockMatrix *mat);

void
BlockMatrix_print_padding(BlockMatrix *mat);


#endif
