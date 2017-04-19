#include "Vector.h"
#include "Block.h"
#include "lapacke.h"

#ifndef _BLOCK_MATRIX_H
#define _BLOCK_MATRIX_H

/* Translates a row,col index to a block format index (column-wise) */
#define POS(i,j,nr_blk_cols) \
((((i) / BLK_LEN) * (nr_blk_cols) * BLK_SIZE) + (((j) / BLK_LEN) * BLK_SIZE) + (((j) % BLK_LEN) * BLK_LEN) + ((i) % BLK_LEN))

typedef struct _BlockMatrix {
    int nr_rows;   // Number of rows
    int nr_cols;   // Number of columns

    int nr_blk_rows;   // Number of block rows
    int nr_blk_cols;   // Number of block columns

    double *data;      // Host data
    double *data_d;    // Device data
} BlockMatrix;

int
BlockMatrix_init(BlockMatrix *mat,
                 int nr_rows,
                 int nr_cols);

int
BlockMatrix_init_constant(BlockMatrix *mat,
	                      int nr_rows,
	                      int nr_cols,
	                      double constant);

/**
 * Initializes a matrix with 0s.
 *
 * @param mat Matrix to initialize.
 * @param nr_rows Number of rows.
 * @param nr_cols Number of columns.
 */
int
BlockMatrix_init_zero(BlockMatrix *mat,
                      int nr_rows,
                      int nr_cols);

int
BlockMatrix_free(BlockMatrix *mat);

/**
 * Initializes a matrix's meta-info but does not allocate
 * memory for its data.
 *
 * @param mat Matrix to initialize info.
 * @paran nr_rows Number of rows.
 * @param nr_cols Number of columns.
 */
int
BlockMatrix_init_info(BlockMatrix *mat,
                      int nr_rows,
                      int nr_cols);

int
BlockMatrix_print(BlockMatrix *mat);

void
BlockMatrix_print_blocks(BlockMatrix *mat);

void
BlockMatrix_print_padding(BlockMatrix *mat);

#ifdef __cplusplus
extern "C"
#endif
int
BlockMatrix_size_bytes(BlockMatrix *mat);

/**
 * Copies data from host to device.
 */
#ifdef __cplusplus
extern "C"
#endif
int
BlockMatrix_copy_host_to_device(BlockMatrix *in);

#ifdef __cplusplus
extern "C"
#endif
int
BlockMatrix_copy_device_to_host(BlockMatrix *in);

/**
 * CudaMalloc device matrix.
 */
#ifdef __cplusplus
extern "C"
#endif
int
BlockMatrix_init_device(BlockMatrix *in);

#ifdef __cplusplus
extern "C"
#endif
int
BlockMatrix_free_device(BlockMatrix *in);

#endif
